import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

"""
This module builds a standard pix2pix image-to-image GAN based on the work of
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 

We aim to recover blurred, compressed and noisy images from dSTORM acquisition 
coming from a cellphone. It accepts image pairs (.png) A-to-B where the images
are concated vertically. 

# HOW TO USE? (command line)


#### BIG DATASE
cd /home/diederich/Documents/STORM/PYTHON/pytorch-CycleGAN-and-pix2pix/

# start server
python -m visdom.server &


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

"""


opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
