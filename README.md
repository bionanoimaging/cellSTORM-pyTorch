<img src='imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>

# cellSTORM based on CycleGAN and pix2pix in PyTorch

This is our ongoing PyTorch implementation of image restauration for dSTORM acquisitions coming from cellphone cameras 

The initial code was written by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung89). Some modification were made by our group. Removing the checkerboard artifact based on [AUGUSTUS ODENA](distill.pub/2016/deconv-checkerboard/) was also implemented in the U-NET generator. 



#### CycleGAN: [[Project]](https://junyanz.github.io/CycleGAN/) [[Paper]](https://arxiv.org/pdf/1703.10593.pdf) [[Torch]](https://github.com/junyanz/CycleGAN)
<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="900"/>

#### Pix2pix:  [[Project]](https://phillipi.github.io/pix2pix/) [[Paper]](https://arxiv.org/pdf/1611.07004v1.pdf) [[Torch]](https://github.com/phillipi/pix2pix)

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="900px"/>

#### [[EdgesCats Demo]](https://affinelayer.com/pixsrv/)  [[pix2pix-tensorflow]](https://github.com/affinelayer/pix2pix-tensorflow)   
Written by [Christopher Hesse](https://twitter.com/christophrhesse)  

<img src='imgs/edges2cats.jpg' width="600px"/>

If you use this code for your research, please cite:

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks  
[Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros)  
In arxiv, 2017. (* equal contributions)  


Image-to-Image Translation with Conditional Adversarial Networks  
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros)   
In CVPR 2017.


## Prerequisites
- Linux or macOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```
- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

### cellSTORM train/test
- Have a look at the MATLAB repo to create the datasets (cellSTORM-MATLAB):

- Train a model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --no_dropout
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`
- Train the model:
```bash
python train.py \
--dataroot /home/diederich/Documents/STORM/DATASET_NN/02_Datapairs/MOV_2018_03_06_11_43_47_randomBlink2500_lines_ISO6400_texp_1_125testSTORM_4000frames_2500emitter_dense_256px_params_png_frames_shifted_combined \
--ndf 32 \
--ngf 32 \
--which_model_netG unet_256 \
--model pix2pix \
--which_direction AtoB \
--dataset_mode aligned \
--norm batch \
--pool_size 0 \
--save_latest_freq 1000 \
--batchSize 4 \
--input_nc 1 \
--output_nc 1 \
--gpu_ids 0,1 \
--loadSize 256 \
--fineSize 256 \
--lr 0.0001  \
--beta1 0.5 \
--display_freq 100 \
--name random_blink_psf_bkgr_nocheckerboard_gtpsf_V5_shifted_UNET_lambda_A_1000_lambda_cGAN_0.5_ISO_6400_random_lines  \
--lambda_A 1000 \
--lambda_cGAN 2 \
--no_lsgan \
```
- Train the model:
```bash
python test.py \
--dataroot /home/diederich/Documents/STORM/DATASET_NN/04_UNPROCESSED_RAW_HW/2018-01-23_17.53.21_oldSample_ISO3200_10xEypiece_texp_1_30_256 \
--ndf 32 \
--ngf 32 \
--which_model_netG unet_256 \
--model pix2pix \
--which_direction AtoB \
--dataset_mode aligned \
--norm batch \
--batchSize 8 \
--input_nc 1 \
--output_nc 1 \
--gpu_ids 0,1 \
--loadSize 256 \
--fineSize 256 \
--name random_blink_psf_bkgr_nocheckerboard_gtpsf_V5_shifted_UNET \
--how_many 100000
```

The test results will be saved to a tiff file here: `./myresults/`.

## Citation
If you use this code for your research, please cite our papers.
```
@article{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  journal={arXiv preprint arXiv:1703.10593},
  year={2017}
}

@article{pix2pix2016,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={arxiv},
  year={2016}
}
```

## Related Projects
[CycleGAN](https://github.com/junyanz/CycleGAN): Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks  
[pix2pix](https://github.com/phillipi/pix2pix): Image-to-image translation with conditional adversarial nets  
[iGAN](https://github.com/junyanz/iGAN): Interactive Image Generation via Generative Adversarial Networks

## Cat Paper Collection
If you love cats, and love reading cool graphics, vision, and learning papers, please check out the Cat Paper Collection:  
[[Github]](https://github.com/junyanz/CatPapers) [[Webpage]](https://people.eecs.berkeley.edu/~junyanz/cat/cat_papers.html)

## Acknowledgments
Code is inspired by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).
