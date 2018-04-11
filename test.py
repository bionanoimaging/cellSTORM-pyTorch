import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import numpy as np
import tifffile
from PIL import Image


"""
This module builds a standard pix2pix image-to-image GAN based on the work of
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 

We aim to recover blurred, compressed and noisy images from dSTORM acquisition 
coming from a cellphone. It accepts image pairs (.png) A-to-B where the images
are concated vertically. 

WARNING: It saves the images to TIFF stacks (BIG TIFF - always concatening the files to the end )

# HOW TO USE? (command line)


cd /home/diederich/Documents/STORM/PYTHON/pytorch-CycleGAN-and-pix2pix/

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



"""

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

# get the image filesize
# options for fine/load-size is prohibitted! 
file_names = [fn for fn in os.listdir(opt.dataroot+'/test')
              if any(fn.endswith(ext) for ext in '.png')][0]
im_size = Image.open(opt.dataroot+'/test/'+file_names).size
opt.loadSize = im_size[1]    
opt.fineSize = im_size[1] 

# accept only grayscale images  
opt.input_nc = 1
opt.output_nc = 1 

# some further dataset settings
opt.which_direction = 'AtoB'




# create filedir according to the filename
dataroot_name = opt.dataroot.split('/')[-1]
myfile_dir = ('./myresults/' + dataroot_name)
if not os.path.exists(myfile_dir):
    os.makedirs(myfile_dir)
    
# create filenames
realA_filename = myfile_dir + '/realA.tiff'
realB_filename = myfile_dir + '/realB.tiff'
fakeB_filename = myfile_dir + '/fakeB.tiff'

# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    
    print('%04d: process image...' % (i))
 
    # realA
    name_realA = visuals.iteritems().next()[0]
    val_realA = visuals.iteritems().next()[1]
    val_realA = np.squeeze(val_realA[:,:,0])
    tifffile.imsave(realA_filename, val_realA, append=True, bigtiff=True)
    
    # fakeB
    name_fakeB = visuals.items()[1][0]
    val_fakeB = visuals.items()[1][1]
    val_fakeB = np.squeeze(val_fakeB[:,:,0])
    tifffile.imsave(fakeB_filename, val_fakeB, append=True, bigtiff=True)
    
    # realB
    name_realB = visuals.items()[2][0]
    val_realB = visuals.items()[2][1]
    val_realB = np.squeeze(val_realB[:,:,0])
    tifffile.imsave(realB_filename, val_realB, append=True, bigtiff=True)



