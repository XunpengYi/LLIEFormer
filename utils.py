import os
import sys
import torch
from tqdm import tqdm
from loss_ssim import SSIM
import cv2
import math
import numpy as np

def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "eval")
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

    train_images_low_path = []
    train_images_high_path = []
    val_images_low_path = []
    val_images_high_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG"]
    train_high_root = os.path.join(train_root, "high")
    train_low_root = os.path.join(train_root, "low")

    val_high_root = os.path.join(val_root, "high")
    val_low_root = os.path.join(val_root, "low")
    train_low_path = [os.path.join(train_low_root, i) for i in os.listdir(train_low_root)
                      if os.path.splitext(i)[-1] in supported]
    train_high_path = [os.path.join(train_high_root, i) for i in os.listdir(train_high_root)
                       if os.path.splitext(i)[-1] in supported]

    val_low_path = [os.path.join(val_low_root, i) for i in os.listdir(val_low_root)
                    if os.path.splitext(i)[-1] in supported]
    val_high_path = [os.path.join(val_high_root, i) for i in os.listdir(val_high_root)
                     if os.path.splitext(i)[-1] in supported]

    assert len(train_low_path) == len(train_high_path), ' The length of train dataset does not match. low:{}, high:{}'.format(
        len(train_low_path), len(train_high_path))
    assert len(val_low_path) == len(val_high_path), ' The length of val dataset does not match. low:{}, high:{}'.format(
        len(val_low_path), len(val_high_path))
    print("image pair check finish")

    for index in range(len(train_low_path)):
        img_low_path = train_low_path[index]
        img_high_path = train_high_path[index]
        train_images_low_path.append(img_low_path)
        train_images_high_path.append(img_high_path)

    for index in range(len(val_low_path)):
        img_low_path = val_low_path[index]
        img_high_path = val_high_path[index]
        val_images_low_path.append(img_low_path)
        val_images_high_path.append(img_high_path)

    total_dataset_nums = len(train_low_path) + len(train_high_path) + len(val_low_path) + len(val_high_path)
    print("{} images were found in the dataset.".format(total_dataset_nums))
    print("{} low light images for training.".format(len(train_low_path)))
    print("{} normal light images for training ref.".format(len(train_high_path)))
    print("{} low light images for validation.".format(len(val_low_path)))
    print("{} normal light images for validation ref.".format(len(val_high_path)))
    return train_low_path, train_high_path, val_low_path, val_high_path


def train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch, batch_size):
    model.train()
    loss_function = torch.nn.L1Loss(reduction='mean')
    loss_function1 = torch.nn.MSELoss(reduction='mean')
    loss_function2 = SSIM(window_size=96)

    if torch.cuda.is_available():
        loss_function = loss_function.to(device)
        loss_function1 = loss_function1.to(device)
        loss_function2 = loss_function2.to(device)
    accu_loss = torch.zeros(1).to(device)
    accu_l1_loss = torch.zeros(1).to(device)
    accu_l2_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        image_low, image_high = data

        if torch.cuda.is_available():
            image_low = image_low.to(device)
            image_high = image_high.to(device)

        pred = model(image_low)

        pred = recover_img(pred)
        image_high = recover_img(image_high)

        loss_L1 = 6 * loss_function(pred, image_high)
        loss_L2 = 4 * loss_function1(pred, image_high)
        loss_ssim = loss_function2(pred, image_high)
        loss = loss_L1 + loss_L2 + loss_ssim
        loss.backward()
        accu_loss += loss.detach()
        accu_l1_loss += loss_L1.detach()
        accu_l2_loss += loss_L2.detach()
        accu_ssim_loss += loss_ssim.detach()
        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  L1 loss: {:.3f}  SSIM loss: {:.3f}  lr: {:.6f}".format(epoch, accu_loss.item() / (step + 1), accu_l1_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), lr)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_l1_loss.item() / (step + 1), accu_l2_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, lr, best_ssim, best_psnr, val_save):
    loss_function = torch.nn.L1Loss(reduction='mean')
    loss_function1 = torch.nn.MSELoss()
    loss_function2 = SSIM(window_size=224)

    model.eval()
    accu_loss = torch.zeros(1).to(device)
    accu_l1_loss = torch.zeros(1).to(device)
    accu_l2_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    psnr_accu = 0
    ssim_accu = 0
    if torch.cuda.is_available():
        loss_function = loss_function.to(device)
        loss_function1 = loss_function1.to(device)
        loss_function2 = loss_function2.to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        image_low, image_high = data
        
        if torch.cuda.is_available():
            image_low = image_low.to(device)
            image_high = image_high.to(device)
            
        pred = model(image_low.to(device))

        pred = recover_img(pred)
        image_high = recover_img(image_high)

        loss_L1 = 6 * loss_function(pred, image_high)
        loss_L2 = 4 * loss_function1(pred, image_high)
        loss_ssim = loss_function2(pred, image_high)
        loss = loss_L1 + loss_L2 + loss_ssim
        accu_loss += loss
        accu_l1_loss += loss_L1
        accu_l2_loss += loss_L2
        accu_ssim_loss += loss_ssim

        if val_save == True:
            output = pred.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            savepic(output, step)

        psnr_accu += PSNR_compute(pred.cpu().detach(), image_high.cpu().detach())
        ssim_accu += SSIM_compute(pred.cpu().detach(), image_high.cpu().detach())

        data_loader.desc = "[val epoch {}] loss: {:.3f}  L1 loss: {:.3f}  SSIM loss: {:.3f}  lr: {:.6f}".format(epoch, accu_loss.item() / (step + 1), accu_l1_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), lr)
    psnr = psnr_accu / (step + 1)
    ssim = ssim_accu / (step + 1)

    if psnr >= best_psnr:
        best_psnr_data = psnr
    else:
        best_psnr_data = best_psnr

    if ssim >= best_ssim:
        best_ssim_data = ssim
    else:
        best_ssim_data = best_ssim

    print("[val epoch: {}] ssim: {:.3f}   psnr: {:.3f}  best_ssim: {:.3f}  best_psnr: {:.3f}".format(epoch, ssim, psnr, best_ssim_data, best_psnr_data))
    return accu_loss.item() / (step + 1), accu_l1_loss.item() / (step + 1), accu_l2_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), ssim, psnr

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def recover_img(x):
    x = x * 0.5 + 0.5
    return x
    
def PSNR_compute(y_input, y_target):
    mse_output = torch.mean((y_input - y_target)**2)
    if mse_output == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse_output))
    
def SSIM_compute(y_input, y_target):
    from skimage.metrics._structural_similarity import structural_similarity as ssim
    _, C, H, W = y_input.shape
    y_input = np.array(y_input)
    y_target = np.array(y_target)
    assert(C == 1 or C == 3)
    # N x C x H x W -> N x W x H x C -> N x H x W x C
    y_input = np.swapaxes(y_input, 1, 3)
    y_input = np.swapaxes(y_input, 1, 2)
    y_target = np.swapaxes(y_target, 1, 3)
    y_target = np.swapaxes(y_target, 1, 2)
    sum_structural_similarity_over_batch = 0.
    if C == 3:
        sum_structural_similarity_over_batch += ssim(
            y_input[0, :, :, :], y_target[0, :, :, :], multichannel=True)
    else:
        sum_structural_similarity_over_batch += ssim(
            y_input[0, :, :, 0], y_target[0, :, :, 0])

    return sum_structural_similarity_over_batch

def savepic(outputpic, name):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    img_name = "./val_img_saver/" + str(name) + ".png"
    cv2.imwrite(img_name, outputpic)