# [ICIP 2023] LLIEFormer: A Low-Light Image Enhancement Transformer Network with a Degraded Restoration Model
### [Paper](https://ieeexplore.ieee.org/abstract/document/10222840) | [Code](https://github.com/xunpengyi/LLIEFormer)

**LLIEFormer: A Low-Light Image Enhancement Transformer Network with a Degraded Restoration Model**
Xunpeng Yi, Yuxuan Wang, Yizhen Zhao, Jia Yan, Weixia Zhang in ICIP 2023

## Pretrained Model
We provide the pretrained models:

- LLIEFormer pretrained on LOL dataset is [here](https://pan.baidu.com/s/15fRZoSGX_8hSgJJTVm5gaQ) (code: 9m5g).

| Method | PSNR | SSIM |
| :-- | :--: | :--: |
| **LLIEFormer** | **22.08** | **0.883** |

 - LLIEFormer pretrained on PairL1.6K dataset is [here](https://pan.baidu.com/s/1D8KkgpAcki1mmAMiTT0tGg) (code: v513) (update version).

| Method | PSNR | SSIM |
| :-- | :--: | :--: |
| Zero-DCE | 16.90 | 0.678 | 
| KinD | 17.27 | 0.645 | 
| KinD++ | 18.52 | 0.701 | 
| LIME | 18.19 | 0.671 | 
| RUAS | 17.91 | 0.633 | 
| RetinexNet | 16.71 | 0.626 | 
| **LLIEFormer** | **25.14** | **0.797** | 

## Test
```bash
cd LLIEFormer-main/
CUDA_VISIBLE_DEVICES=0 python test.py
```

## Train
```bash
cd LLIEFormer-main/
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Citation
If you find our work useful for your research, please cite our paper
```
@inproceedings{yi2023llieformer,
  title={Llieformer: A Low-Light Image Enhancement Transformer Network with a Degraded Restoration Model},
  author={Yi, Xunpeng and Wang, Yuxuan and Zhao, Yizhen and Yan, Jia and Zhang, Weixia},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  pages={1195--1199},
  year={2023},
  organization={IEEE}
}
```
