import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import cv2

# from engine import train_one_epoch, evaluate
# import utils
# import transforms as T
#
# def get_transform(train):
#     transforms = []
#     # converts the image, a PIL image, into a PyTorch Tensor
#     transforms.append(T.ToTensor())
#     # if train:
#     #     # during training, randomly flip the training images
#     #     # and ground-truth for data augmentation
#     #     transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)

class SW_SEG_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, GT_dir, transform=None):
        self.data_dir = data_dir
        self.GT_dir = GT_dir
        self.transform = transform
        # load all image files, sorting them to
        # ensure that they are aligned

        #### hoseo_AAA
        self.subjects = [name for name in os.listdir(self.data_dir)
                         if os.path.isdir(self.data_dir) and not name.startswith('.')]
        self.subjects = sorted(self.subjects)
        img_path=[]
        mask_path=[]
        # print(self.subjects)
        for sub in self.subjects:
            img_path.extend([os.path.join(sub, name)
                             for name in os.listdir(os.path.join(self.data_dir,sub))
                             if name.endswith(".png")])
            mask_path.extend([os.path.join(sub,name)
                              for name in os.listdir(os.path.join(self.GT_dir,sub))
                              if name.endswith("png")])

        self.imgs = sorted(img_path)
        self.masks = sorted(mask_path)
        ####

        # self.imgs = list(sorted(os.listdir(os.path.join(root, "raw"))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))

    def __getitem__(self, idx):
        # load images ad masks

        #### hoseo_AAA
        img_path = os.path.join(self.data_dir, self.imgs[idx])
        mask_path = os.path.join(self.GT_dir, self.masks[idx])
        # print(img_path)
        # print(mask_path)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        mask = Image.open(mask_path)
        mask = np.array(mask)
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            # transformed = self.transform(image=img)
            img = transformed['image']
            mask = transformed['mask']
            if np.isnan(img).any() or np.isnan(mask).any():
                print("---------------NAN is occurred-------------")

        mask = np.array(mask) / 255
        mask = mask.astype(np.uint8)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        # print(masks[0].shape)
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1]) - 5
            xmax = np.max(pos[1]) + 5
            ymin = np.min(pos[0]) - 5
            ymax = np.max(pos[0]) + 5
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = np.float32(0.0)

        if num_objs > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            area = torch.zeros(0, dtype=torch.float32)

        # suppose all insttances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # print(target)


        return img, target

    def __len__(self):
        return len(self.imgs)