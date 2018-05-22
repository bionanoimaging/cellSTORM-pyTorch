import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import skvideo.io
import matplotlib as plt
import numpy as np


class VideoDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir_AB = opt.dataroot
        
        # open videoreader
        self.videogen = skvideo.io.vreader(self.dir_AB )
        
        # define roisize and center where each frame will be extracted
        self.roisize = opt.roisize #512
        self.padwidth = opt.padwidth
        self.xcenter = opt.xcenter
        self.ycenter =  opt.ycenter
        

    def __getitem__(self, index):
        # assign dummy variable
        AB_path = 'video'
        
        # read frame
        frame = self.videogen.next()

        # if no center is chosen, select the videos center
        if self.xcenter == -1:
            self.xcenter = int(frame.shape[0]/2)
            print('New xcenter: ' + str(self.xcenter))
        if self.ycenter == -1:
            self.ycenter = int(frame.shape[1]/2)
            print('New ycenter: ' + str(self.ycenter))
        if self.roisize == -1:
            self.roisize = int(np.min(frame.shape[0:1]))
            print('New roisize: ' + str(self.roisize))
        
        

        # crop frame to ROI
        frame_crop = frame[self.xcenter-self.roisize/2:self.xcenter+self.roisize/2, self.ycenter-self.roisize/2:self.ycenter+self.roisize/2,:]
        
        npad = ((self.padwidth, self.padwidth), (self.padwidth, self.padwidth), (0, 0))
        frame_pad = np.pad(frame_crop, npad, 'reflect')

        # convert from NP to PIL
        A = Image.fromarray(frame_pad).convert('RGB')
        A = transforms.ToTensor()(A)

        # normalize 
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)

        tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        A = tmp.unsqueeze(0)
        B = A


        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        videometadata = skvideo.io.ffprobe(self.dir_AB)
        #print(videometadata)
        #print(self.dir_AB)
        num_frames = np.int(videometadata['video']['@nb_frames'])

        return num_frames

    def name(self):
        return 'VideoDataset'
