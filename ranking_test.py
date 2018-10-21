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

def display_data(gallery,target,i):
    ims = []
    t_im = cv2.imread(target["path"])
    cv2.rectangle(t_im,(0,0),(t_im.shape[0],t_im.shape[0]),(0,255,0),3)
    ims.append(t_im)
    
    for e in gallery:
        ims.append(cv2.imread(e["path"]))
    numpy_horizontal = np.hstack(ims)
    #cv2.imwrite("sameness_out/"+str(i)+".png", numpy_horizontal)
    cv2.imshow('Main', numpy_horizontal)
    cv2.waitKey(0)
def get_distance(e1, e2):
    
    #print(np.array(globs).shape)
    e1 = np.concatenate((e1["features"][0].flatten(),e1["features"][1].flatten()))
    e2 = np.concatenate((e2["features"][0].flatten(),e2["features"][1].flatten()))
    #e1 = e1["features"][0].flatten()
    #e2 = e2["features"][0].flatten()
    return np.linalg.norm(e1 - e2)

Xf = open('data','rb')
data = np.array(pickle.load(Xf))
Xf.close()

same = {}
for e in data:
    if e["id"] not in same:
        same[e["id"]] = []
    same[e["id"]].append(e)
   
hits = 0
ids = list(same.keys())
n_steps = 10
hit_dist, miss_dist = [],[]
for j in range(n_steps):
    
    rand_init = []
    
    target_group = same[random.choice(ids)]
    target_id = random.randint(0,len(target_group)-1)
    target = target_group[target_id]
    #print(len(target_group))
    gallery = []
    for i in range(len(target_group)):
        if i != target_id:
            gallery.append(target_group[i])
    new_data = []
    for e in data:
        if e["id"] != target["id"]:
            new_data.append(e)
            
    n_samples = min(len(new_data),1000)
    gallery.extend(random.sample(list(new_data), n_samples))
    gallery.sort(key=lambda e: get_distance(e, target))
    top5 = []
    for i in range(5):
        top5.append(gallery[i]["id"])
    if top5[0] == target["id"]:# or get_distance(gallery[0], target) < 0.7:
        hit_dist.append(get_distance(gallery[0], target))
        hits+=1
    else:
        miss_dist.append(get_distance(gallery[0], target))
    display_data(gallery[0:10],target,j)   
print(hits/n_steps)
print( "hit: ", np.mean(hit_dist))
print("miss: ",np.mean(miss_dist))
plt.hist(hit_dist, bins='auto')  # arguments are passed to np.histogram
plt.hist(miss_dist, bins='auto')  # arguments are passed to np.histogram

plt.title("Hits")
plt.show()

#plt.hist(miss_dist, bins='auto')  # arguments are passed to np.histogram
#plt.title("Misses")
#plt.show()
