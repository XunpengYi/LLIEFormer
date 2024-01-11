import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from my_dataset import MyDataSet
from LLIEFormer_LA import LLIEFormer as create_model
from utils import read_data, train_one_epoch, evaluate, create_lr_scheduler
import datetime
import transforms as T

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    best_ssim = 0
    best_psnr = 0

    train_images_path, train_images_lpath, val_images_path, val_images_lpath = read_data(args.data_path)

    val_save_flag = args.val_save
    img_size = 160
    data_transform = {
        "train": T.Compose([T.RandomCrop(img_size),
                            T.RandomHorizontalFlip(0.5),
                            T.RandomVerticalFlip(0.5),
                            T.ToTensor(),
                            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),

        "val": T.Compose([T.ToTensor(),
                            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_lpath=train_images_lpath,
                              transform=data_transform["train"],
                              split="train")

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_lpath=val_images_lpath,
                            transform=data_transform["val"],
                            split="val")

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model().to(device)
    if args.use_dp == True:
        model = torch.nn.DataParallel(model).cuda()

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.val_save:
        if os.path.exists("./val_img_saver") is False:
            os.makedirs("./val_img_saver")

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-3)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.epochs):
        # train
        train_loss, train_L1, train_L2, train_ssim, lr = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                lr_scheduler=lr_scheduler,
                                                data_loader=train_loader,
                                                device=device,
                                                batch_size=batch_size,
                                                epoch=epoch)

        # validate
        val_loss, val_L1, val_L2, val_ssim, current_ssim, current_psnr = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,lr=lr,
                                     best_ssim=best_ssim, best_psnr=best_psnr,
                                     val_save=val_save_flag)

        tb_writer.add_scalar("train_total_loss", train_loss, epoch)
        tb_writer.add_scalar("train_L1_loss", train_L1, epoch)
        tb_writer.add_scalar("train_L2_loss", train_L2, epoch)
        tb_writer.add_scalar("train_ssim_loss", train_ssim, epoch)
        tb_writer.add_scalar("val_total_loss", val_loss, epoch)
        tb_writer.add_scalar("val_L1_loss", val_L1, epoch)
        tb_writer.add_scalar("val_L2_loss", val_L2, epoch)
        tb_writer.add_scalar("val_ssim_loss", val_loss, epoch)
        tb_writer.add_scalar("current_ssim", current_ssim, epoch)
        tb_writer.add_scalar("current_psnr", current_psnr, epoch)

        #save the best model in the eval dataset
        if current_psnr >= best_psnr:
            if args.use_dp == True:
                save_file = {"model": model.module.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            else:
                save_file = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            torch.save(save_file, "./weights/best_psnr_model.pth")
            best_psnr = current_psnr

        if current_ssim >= best_ssim:
            if args.use_dp == True:
                save_file = {"model": model.module.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            else:
                save_file = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
            torch.save(save_file, "./weights/best_ssim_model.pth")
            best_ssim = current_ssim

        if args.use_dp == True:
            save_file = {"model": model.module.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
        else:
            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch,
                         "args": args}
        torch.save(save_file, "./weights/final_model.pth")

        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {train_loss:.4f}\n" \
                         f"val_loss: {val_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"current_ssim: {current_ssim:.4f}\n" \
                         f"current_psnr: {current_psnr:.4f}\n"\
                         f"best_ssim: {best_ssim:.4f}\n"\
                         f"best_psnr: {best_psnr:.4f}\n"
            f.write(train_info + "\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--data-path', type=str,
                        default="./dataset/LOLdataset")
    
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--val-save', default=True, help='save the val process image')
    parser.add_argument('--use-dp', default=False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
