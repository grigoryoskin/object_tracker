"""Train with optional Global Distance, Local Distance, Identification Loss."""
from __future__ import print_function

import sys
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse
import PIL
from aligned_reid.model.Model import Model
from aligned_reid.model.TripletLoss import TripletLoss
from aligned_reid.model.loss import global_loss
from aligned_reid.model.loss import local_loss

from aligned_reid.utils.utils import *

from  aligned_reid.utils import distance
import time

def get_distance(e1, e2):
    
    
    e1 = np.concatenate((e1[0].flatten(),e1[1].flatten()))
    e2 = np.concatenate((e2[0].flatten(),e2[1].flatten()))
    #e1 = e1[0].flatten()
    #e2 = e2[0].flatten()
    return np.linalg.norm(e1 - e2)
    

def preprocess_image(im):
  im = im.resize((128,256))
  #im.save('tmp.jpg')
  I = np.asarray(im)
  I = I - np.mean(I)
  I = I/np.std(I)
  I= I.transpose(2, 0, 1)
  return I



def main():
  
  TVT, TMO = set_devices((-1,))

 

  model = Model(local_conv_out_channels=128,
                num_classes=751)
 
  model_w = DataParallel(model)

  
  map_location = (lambda storage, loc: storage)
  sd = torch.load('ckpt.pth', map_location=map_location)
  load_state_dict(model, sd)
  
  print('Loaded model weights')
  ims = []
  
  im1 = PIL.Image.open('demo/0.jpg')
  ims.append(preprocess_image(im1))
  im2 = PIL.Image.open('demo/1.jpg')
  ims.append(preprocess_image(im2))
  
  ims = Variable(TVT(torch.from_numpy(np.array(ims)).float()))
  start = time.time()
  global_feat, local_feat = model(ims)[:2]
  global_feat = np.vstack(global_feat.data.cpu().numpy())
  local_feat = local_feat.data.cpu().numpy()
  distance = get_distance((global_feat[0],local_feat[0]),(global_feat[1],local_feat[1]))
  print("distance: ", distance)
 
 


if __name__ == '__main__':
  main()
