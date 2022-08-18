import numpy as np
from sklearn.cluster import KMeans
import cv2
import os

from vol_eval import *

save_path = "./result"
mia_probability_path = "./result_archieve/mia_probabiblity"
# mia_probability_path = "./result_archieve/mia_crop_probability"

subjects = os.listdir(mia_probability_path)
subjects = sorted(subjects)
print(subjects)
print(len(subjects))


for subject in subjects:
    os.mkdir(os.path.join(save_path,subject))
    write_path = os.path.join(save_path,subject)
    pm_list = os.listdir(os.path.join(mia_probability_path,subject))
    pm_list = sorted(pm_list)

    pm_3d_list = []
    for pm_name in pm_list:
        pm_path = os.path.join(mia_probability_path,subject,pm_name)
        pm = np.load(pm_path)
        pm_3d_list.append(pm)

    pm = np.stack(pm_3d_list)
    # origin = pm.copy()
    # print(subject, pm.shape)

    # for i in range(512):
    for i in range(256):
        pm[:,:,i] = cv2.GaussianBlur(pm[:,:,i],(9,9),0.3)

    # print(origin==pm)
    #######
    print(subject, pm.shape)
    t,w,h = pm.shape
    pm = pm.flatten()


    kmean = KMeans(n_clusters=6)
    kmean.fit(pm.reshape(-1,1))

    print(kmean.labels_)
    print(kmean.cluster_centers_)
    print(min(kmean.cluster_centers_))
    print(len(kmean.labels_))

    min_label = min(kmean.cluster_centers_)
    min_label = kmean.cluster_centers_.tolist().index(min_label)

    for i in range(len(kmean.labels_)):
        # print(kmean.labels_[i], min_label)
        if kmean.labels_[i] == min_label:
            # if pm[i]!=0:
            #     print("OCCUR 0")
            pm[i]=0

    pm = pm.reshape((t,w,h))

    #########
    threshold = 0.5
    for i in range(t):
        img_name = pm_list[i]
        img_name = img_name.split('.')[0]
        write_name = os.path.join(write_path, img_name)
        # print(write_name)

        mask = pm[i,:,:]
        mask = np.where(mask>=threshold, 1.0, 0.0)
        mask *= 255
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
        img_mask = np.array(mask)



        cv2.imwrite(write_name+".png", img_mask)


total_ol = []
total_ja = []
total_di = []
total_fp = []
total_fn = []

for subject in subjects:
    overlap, jaccard, dice, fn, fp = eval_volume_from_mask(subject, pred_path = "./result")
    # overlap, jaccard, dice, fn, fp = eval_volume_from_mask(subject, pred_path = "./result", GT_path="./dataset/ROI_crop_pos")
    print(subject +' overlap: %.4f jaccard: %.4f dice: %.4f fn: %.4f fp: %.4f' % (
    overlap, jaccard, dice, fn, fp))
    total_ol.append(overlap)
    total_ja.append(jaccard)
    total_di.append(dice)
    total_fn.append(fn)
    total_fp.append(fp)
# print(subject, overlap, jaccard, dice, fn, fp)
print('[ Average volume evaluation] overlap: %.4f jaccard: %.4f dice: %.4f fn: %.4f fp: %.4f' % (
    np.mean(total_ol), np.mean(total_ja), np.mean(total_di), np.mean(total_fn), np.mean(total_fp)))