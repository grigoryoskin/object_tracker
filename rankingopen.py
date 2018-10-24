from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg
import pickle
from  aligned_reid.utils import distance
import cv2
import random
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import matplotlib.pyplot as plt

def get_distance(e1, e2):
    
    #print(np.array(globs).shape)
    #e1 = np.concatenate((e1["features"][0].flatten(),e1["features"][1].flatten()))
    #e2 = np.concatenate((e2["features"][0].flatten(),e2["features"][1].flatten()))
    e1 = e1["features"][0].flatten()
    e2 = e2["features"][0].flatten()
    return np.linalg.norm(e1 - e2)
    #return loc_dist[0][0] , glob_dist[0][0]

Xf = open('data','rb')
data = np.array(pickle.load(Xf))
Xf.close()
print("data size: ", len(data))

print("feature shape: ", data[0]["features"][0].shape)
same = {}
for e in data:
    if e["id"] not in same:
        same[e["id"]] = []
    same[e["id"]].append(e)
   
hits = 0
ids = list(same.keys())
print("elements per id: ", len(data)/len(ids))

###################################VARIABLES#######
n_steps = 1000
thresh = 0.85
n_samples = 100
####################################################

hit_dist, miss_dist = [],[]

##############################################################
for j in range(int(n_steps/2)):
    rand_init = []
    
    target_group = same[random.choice(ids)]
    target_id = random.randint(0,len(target_group)-1)
    target = target_group[target_id]
    #print(len(target_group))
    gallery = []
    new_data = []
    for e in data:
        if e["id"] != target["id"]:
            new_data.append(e)
    
    
    gallery.extend(random.sample(list(new_data), n_samples))
    random.shuffle(gallery)
    miss_d = [get_distance(v, target) for v in gallery[0:5]]
    
    gallery.sort(key=lambda e: get_distance(e, target))
    sorted_miss_d = [get_distance(v, target) for v in gallery[0:5]]
    ratio = np.mean(sorted_miss_d)/np.mean(miss_d)
    miss_dist.append(ratio)
    if ratio > thresh:
        hits+=1
    
for j in range(int(n_steps/2)):
    
    rand_init = []
    

    target_group = same[random.choice(ids)]
    target_id = random.randint(0,len(target_group)-1)
    target = target_group[target_id]
    
    gallery = []
    for i in range(len(target_group)):
        if i != target_id:
            gallery.append(target_group[i])
    new_data = []
    for e in data:
        if e["id"] != target["id"]:
            new_data.append(e)
    
    
    gallery.extend(random.sample(list(new_data), n_samples))
    
    random.shuffle(gallery)
    
    gal_dist = [get_distance(v, target) for v in gallery[0:5]]
    gallery.sort(key=lambda e: get_distance(e, target))
    sorted_gal_dist = [get_distance(v, target) for v in gallery[0:5]]
    ratio = np.mean(sorted_gal_dist)/np.mean(gal_dist)
    hit_dist.append(ratio)
    if ratio < thresh:
        hits+=1
    
print("accuracy: ", hits/n_steps)
plt.hist(hit_dist, bins='auto')  # arguments are passed to np.histogram
plt.hist(miss_dist, bins='auto')  # arguments are passed to np.histogram

plt.title("Hits")
plt.show()

