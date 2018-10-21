"""Train with optional Global Distance, Local Distance, Identification Loss."""
from __future__ import print_function

import sys
sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
import pickle

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





def main():
  
  TVT, TMO = set_devices((-1,))

 

  model = Model(local_conv_out_channels=128,
                num_classes=751)
  # Model wrapper
  model_w = DataParallel(model)

  
  map_location = (lambda storage, loc: storage)
  sd = torch.load('ckpt.pth', map_location=map_location)
  load_state_dict(model, sd)
  
  print('Loaded model weights')
  IMAGES_PATH = "galery/"
  files = os.listdir(IMAGES_PATH)
  files.sort();
  data = []
  formats = ["png", "jpg"]
  curr_id = 0
  ids = {}
  for image_name in files:
    if image_name.split('.')[1] in formats and int(image_name.split('_')[0]) > 0:
      id = image_name.split('_')[0]
      '''
      if id not in ids:
        ids[id] = 0
      else:
        if ids[id] > 20:
          continue
        
      ids[id]+=1
      '''
      im = PIL.Image.open(IMAGES_PATH + image_name)
      im = im.resize((128,256))
      #im.save('tmp.jpg')
      I = np.asarray(im)
      I = I - np.mean(I)
      I = I/np.std(I)
      I= np.expand_dims(I.transpose(2, 0, 1),axis = 0)
      ims = Variable(TVT(torch.from_numpy(I).float()))
      global_feat, local_feat = model(ims)[:2]
      global_feat = np.vstack(global_feat.data.cpu().numpy())
  
      local_feat = local_feat.data.cpu().numpy()   
      
      #print(features)
      print("processing file: ", image_name)
      data.append({"id" : id, "features" : (global_feat, local_feat),"path" : IMAGES_PATH + image_name})
  filename = 'data'
  outfile = open(filename,'wb')
  pickle.dump(data,outfile)
  outfile.close()
 


if __name__ == '__main__':
  main()
