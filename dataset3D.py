import numpy as np
import math
import os
from skimage.util import view_as_windows
import scipy.io as sio
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
import nibabel as nib
import hdf5storage

class Train_dataset(object):
    def __init__(self, batch_size, overlapping=1):
        print('*****__init__*****')
        self.batch_size = batch_size
        self.overlapping = overlapping
        self.data_path = 'select your path'      
        self.subject_list = os.listdir(self.data_path)
        print('len(self.subject_list)',len(self.subject_list))     

    def patches_true(self, iteration):
        subjects_true = self.data_true(iteration)
        patches_true = np.empty([8,64,64,64,1])
        i = 0
        for subject in subjects_true:
            patch = view_as_windows(subject, window_shape=(64,64, 64),step=(16,16,16))
            for d in range(patch.shape[0]):
                for v in range(patch.shape[1]):
                    for h in range(patch.shape[2]):

                        p = patch[d, v, h, :]

                        p = p[:, np.newaxis]

                        p = p.transpose((0, 2, 3, 1))

                        patches_true[i] = p

                        i = i + 1


        return patches_true
        

    def data_true(self, iteration):

        subject_batch = self.subject_list[iteration * self.batch_size:self.batch_size + (iteration * self.batch_size)]
        subjects = np.empty([self.batch_size, 80, 80, 80])
        
        i = 0
        for subject in subject_batch:
           
            if subject != 'Untitled.m':
                filename = os.path.join(self.data_path, subject)

                proxy  = hdf5storage.loadmat(filename)
                
                mat_img = np.array(proxy["temp"])
        
                paddwidthr = int((80 - mat_img.shape[0]) / 2)
                paddheightr = int((80 - mat_img.shape[1]) / 2)
                paddepthr = int((80 - mat_img.shape[2]) / 2)

                if (paddwidthr * 2 + mat_img.shape[0]) != 80:
                    paddwidthl = paddwidthr + 1
                else:
                    paddwidthl = paddwidthr
          

                if (paddheightr * 2 + mat_img.shape[1]) != 80:
                    paddheightl = paddheightr + 1
                else:
                    paddheightl = paddheightr
                

                if (paddepthr * 2 + mat_img.shape[2]) != 80:
            
                    paddepthl = paddepthr + 1
                else:
                    paddepthl = paddepthr
              

                data_padded = np.pad(mat_img,
                                     [(paddwidthl, paddwidthr), (paddheightl, paddheightr), (paddepthl, paddepthr)],
                                     'constant', constant_values=0)
                subjects[i] = data_padded  # remove background
          
                subjects = subjects[:, np.newaxis]
            
                subjects = subjects.transpose((0, 2, 3, 4,1))
             
                i = i + 1
          
        return subjects