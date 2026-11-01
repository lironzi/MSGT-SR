import numpy as np
import tifffile
from PIL import Image
import torch
from torch.utils.data import Dataset
#from utils_misc import rot_mat,check_rotation,rotation
from glob import glob
import torchvision
import pickle
import random
import os
import cv2
from misc.utils_misc import get_new_point,get_new_shape,get_y_boundaries,get_x_bounderies,get_y_bounderies
import torchvision.transforms.functional as F


class ThermalVisDataset(Dataset):
    def __init__(self, dataPath, scale_factor=8,HR_downsample=0,original_size=(256,336),LR_upsample=0,Sessions=[],numPermutationPerImage=4000):
        """
        :param dataPath: the path to the data folder subfolders include the Sentinel tiff images and the numpy thermal image
        :param scale_factor: factor for upscale
        :param HR_downsample: initial downsample the result of this will be the GT of the image
        :param LR_upsample: 1 cause initial upsample of the LR image to HR dimension. 0 otherwise
        :param selected_bands: the bands from sentinel 2 satellite for exp [1,2,3] is RGB
        """

        self.scale_factor = scale_factor
        self.dataPath=dataPath
        self.HR_downsample=HR_downsample
        self.LR_upsample=LR_upsample
        self.numPermutationPerImage = numPermutationPerImage
        self.original_size=original_size
        self.sessions=Sessions
        self.rng = random.Random(0)
        if not Sessions:
            raise ValueError("Expected a non-empty Session list")
        self._loadImgs()
        self._getStats()
        self._precompute_crops()

    def _getStats(self):
        self.stats={}
        for s in self.sessions:
            self.stats[s]=[]
            # visible bands occupy band
            for i in range(4):
                self.stats[s].append((np.mean(self.ThermalVisImgs[s][i]),np.std(self.ThermalVisImgs[s][i])))

    def _loadImgs(self):
        self.ThermalVisImgs={}
        for s in self.sessions:
            self.ThermalVisImgs[s]=[]
            # visible bands occupy bands 1-3 in BGR order
            for i in range(2, 5):
                imgBand = tifffile.imread(os.path.join(self.dataPath, s, 'band_' + str(i) + '.tiff'))
                self.ThermalVisImgs[s].append(imgBand)

            imgBand = np.load(os.path.join(self.dataPath, s, "mosaic_pan.npy"))
            self.ThermalVisImgs[s].append(imgBand)

    def _precompute_crops(self):
        """Generate deterministic crop positions per session image."""
        self.cropPositions={}
        for s in self.sessions:
            h,w = self.ThermalVisImgs[s][0].shape
            for p in range(self.numPermutationPerImage):

                #get a random angle of rotation
                theta = self.rng.uniform(0,360)

                # calculate the new width and height of the rotated image
                rot_w, rot_h = get_new_shape((w, h), theta)

                # calculate the minimum x valid for a crop of height target_height in the coordinate system of the rotated image
                x_min = get_x_bounderies(theta, target_height=self.original_size[0]) + 1

                # random x coordinate of the upper left corner of the crop
                x_rot = x_min + int(self.rng.random() * (rot_w - self.original_size[1] - 2 * x_min))

                # random y coordinate of the upper left corner of the crop by calculating valid max and min values
                y_min_0,y_max_0=get_y_bounderies((w,h),(rot_w,rot_h),theta,x_rot)
                y_min_1, y_max_1 = get_y_bounderies((w, h), (rot_w, rot_h), theta, x_rot+self.original_size[1])
                y_min=np.max([y_min_0,y_min_1])
                y_max=np.min([y_max_0,y_max_1])

                y_rot=int(y_min+self.rng.random()*(y_max-y_min-self.original_size[0]))
                if s not in self.cropPositions.keys():
                    self.cropPositions[s]=[]
                self.cropPositions[s].append((int(x_rot),y_rot,theta))

    def setSeed(self,seed):
        self.rng=random.Random(seed)
        self._precompute_crops()

    def __len__(self):
        return len(self.sessions)*self.numPermutationPerImage

    def __getitem__(self, idx):
        # figure out which session to take
        session_idx = np.mod(idx,len(self.sessions))
        session = self.sessions[session_idx]

        # artificial index based on number of permutation per session
        img_idx = np.mod(idx , self.numPermutationPerImage)
        cropPosition = self.cropPositions[session][img_idx]

        thermalImage = torch.from_numpy(self.ThermalVisImgs[session][-1].astype(float)).unsqueeze(0)

        visImage = torch.from_numpy(np.stack(self.ThermalVisImgs[session][0:3]).astype(float))

        #rotating the image
        rotatedThermal = F.rotate(thermalImage, cropPosition[-1], interpolation=F.InterpolationMode.BILINEAR, expand=True)
        rotatedVis = F.rotate(visImage, cropPosition[-1], interpolation=F.InterpolationMode.BILINEAR, expand=True)

        #cropping the image base on _precompute_crops method
        croppedThermal = rotatedThermal[:,cropPosition[1]:cropPosition[1]+self.original_size[0],cropPosition[0]:cropPosition[0]+self.original_size[1]]
        croppedVis = rotatedVis[:,cropPosition[1]:cropPosition[1]+self.original_size[0],cropPosition[0]:cropPosition[0]+self.original_size[1]]

        if self.HR_downsample:
            croppedThermal = torch.nn.functional.interpolate(input=croppedThermal.unsqueeze(0),scale_factor=1/self.HR_downsample,mode='bicubic',antialias=True).squeeze(0)
            croppedVis = torch.nn.functional.interpolate(input=croppedVis.unsqueeze(0),scale_factor=1/self.HR_downsample,mode='bicubic',antialias=True).squeeze(0)


        #Normalize the images to 0 mean and std of 1
        HR=(croppedThermal-self.stats[session][-1][0])/self.stats[session][-1][1]

        visMean = torch.tensor([m for m, _ in self.stats[session][0:3]]).view(3,1,1)
        visSTD = torch.tensor([std for _,std in self.stats[session][0:3]]).view(3,1,1)
        Vis=(croppedVis-visMean)/visSTD

        #Flip Horizontally
        if self.rng.random()>0.5:
            HR=torch.flip(HR,dims=[2])
            Vis=torch.flip(Vis,dims=[2])

        #Flip Vertically
        if self.rng.random()>0.5:
            HR=torch.flip(HR,dims=[1])
            Vis=torch.flip(Vis,dims=[1])

        LR=torch.nn.functional.interpolate(input=HR.unsqueeze(0),scale_factor=1/self.scale_factor,mode='bicubic',antialias=True).squeeze(0)
        if self.LR_upsample:
          LR=torch.nn.functional.interpolate(input=LR.unsqueeze(0),scale_factor=self.scale_factor,mode='bicubic').squeeze(0)
        return LR,HR,Vis


if __name__=='__main__':
    a = ThermalVisDataset(dataPath="Data",scale_factor=8,HR_downsample=0,original_size=(256,336),LR_upsample=0,Sessions=["111022_1"],numPermutationPerImage=100)
    q=a[1]

