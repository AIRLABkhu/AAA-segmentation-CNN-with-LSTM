import torchvision
import torch
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2
import segmentation_models_pytorch as smp

from SW_SEG_dataset import *

from models.unet_model import UNet
from models.mia import MIA

def get_instance_unet_model(n_channels, n_classes):
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=False)
    return model

def get_instance_unetPlusPlus_model():
    model = smp.UnetPlusPlus(in_channels=3, classes=1)
    return model

def get_instance_mia_model():
    model = MIA()
    return model


def seg_pm_generator(model_name, model_path, dir_path, data_path, subjects, save_image=False, test_mode=False):

    model_path = os.path.join(model_path, model_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # our dataset has two classes only - background and ...


    # get the model using our helper function
    seg_model = get_instance_unet_model(n_channels=3, n_classes=1)
    # move model to the right device
    seg_model.to(device)
    # print(model_path)
    seg_model.load_state_dict(torch.load(model_path))

    test_transform = A.Compose(
        [
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    dataset = SW_SEG_dataset(data_path, './dataset/ROI')

    subject_slice = dict()
    split= 0
    # print(dataset.subjects[13], dataset.subjects[25])
    total_indicies = list(range(0, len(dataset.imgs)))
    for subject in dataset.subjects:
        # print(subject)
        # print([path for path in dataset.imgs if subject in path])
        num_slice = len([path for path in dataset.imgs if subject in path])
        subject_slice[subject] = [split, split + num_slice]
        split += num_slice
    test_indices = list(range(subject_slice[dataset.subjects[subjects[0]]][0],
                          subject_slice[dataset.subjects[subjects[1]]][1]))

    if test_mode:
        train_indices = test_indices
    else:
        train_indices = [index for index in total_indicies if index not in test_indices]
    dataset = torch.utils.data.Subset(dataset, train_indices)
    dataset.dataset.transform = test_transform

    ## for test
    max_val = 0
    min_val = 255
    sigmoid = torch.nn.Sigmoid()

    ## probability map creation
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    for idx, (image, target) in enumerate(dataset):
        image_name = dataset.dataset.imgs[train_indices[idx]]
        # print(image_name)
        subject = image_name.split('/')[0]
        img_num = image_name.split('/')[1]
        write_path = os.path.join(dir_path, subject)
        if not os.path.isdir(write_path):
            os.mkdir(write_path)

        write_name = os.path.join(write_path, img_num)
        if np.isnan(image).any():
            print("imgae have NAN!")
        seg_model.eval()
        with torch.no_grad():
            # prediction = seg_model([image.to(device)])
            prediction = seg_model(image.unsqueeze(0).to(device))
        # im.show()
        if save_image:
            if len(prediction) > 0:
                prediction = sigmoid(prediction)
                pm = prediction.squeeze()
                pm = pm.cpu().numpy()
                pm *= 255
                pm = pm.astype(np.uint8)
                mask = np.array(Image.fromarray(pm))
            else:
                mask = np.zeros((512,512))

            # cv2.imshow("mask predict", np.array(mask))
            # cv2.waitKey(0)
            cv2.imwrite(write_name, mask)

        else:
            if len(prediction) > 0:
                prediction = sigmoid(prediction)
                pm = prediction.squeeze()
                pm=np.array(pm.cpu().numpy(), dtype=np.float32)
                # print(pm.shape)
                # print(pm.dtype)
                # print(np.max(pm), np.min(pm), np.average(pm))
            else:
                 pm = np.zeros((512,512))

            ### save pm
            np.save(write_name,pm)

        # for test
        if max_val < np.max(pm):
            max_val = np.max(pm)
        if min_val > np.min(pm):
            min_val = np.min(pm)

    del seg_model
    # print("Max val : %d  //  Min val : %d" %(max_val, min_val))
    # print(prediction)
    # print("======")
    # print("== Probability map creation is done ==")
    ###
    # probability map is done