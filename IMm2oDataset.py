import os
from os.path import join as PJ
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as T

def check_pos(data_path, pos_path='./dataset/ROI_pos'):
    # print(data_path.split('/'))
    _,_,_,subject,name = data_path.split('/')
    pos_list = os.listdir(PJ(pos_path, subject))
    # print(name)
    # print(pos_list)
    if name in pos_list:
        return True
    else:
        return False


class IMm2oDataset(torch.utils.data.Dataset):
    # max_slice = None => Whole subject slice as a volume

    def __init__(self, data_path, m_length, only_pos = True, pm_path = None, transform=None):

        self.data_path = data_path
        self.m_length = m_length
        self.mask_volumes = []
        self.img_volumes = []
        self.pm_path = pm_path
        self.transform = transform

        self.img_subject_vol_index = {}

        img_subjects = [name for name in os.listdir(data_path)
                        if os.path.isdir(os.path.join(data_path, name))]
        self.subjects = sorted(img_subjects)
        img_index = 0

        for subject in self.subjects:
            images_path = [PJ(data_path, subject, img_path)
                           for img_path in os.listdir(PJ(data_path, subject))
                           if img_path.endswith('png')]
            images_path = sorted(images_path)
            masks_path = [PJ("./dataset", "ROI_pos", subject, img_path)
                          for img_path in os.listdir(PJ('./dataset/ROI_pos', subject))
                          if img_path.endswith('png')]
            masks_path = sorted(masks_path)
            idx_list = []
            for mask_path in masks_path:

                #check Positive
                if (only_pos and check_pos(mask_path)) or not only_pos:
                    # print(mask_path)
                    self.mask_volumes.append(mask_path)
                    img_name = os.path.basename(mask_path).replace('_ROI','')
                    img_name = PJ(data_path,subject, img_name)
                    center = images_path.index(img_name)
                    # print(images_path[center-m_length+1 : center+m_length])
                    self.img_volumes.append(images_path[center-self.m_length+1 : center+self.m_length])
                    idx_list.append(img_index)
                    img_index+=1


            self.img_subject_vol_index[subject]=idx_list


        # print(self.mask_volumes)
        # print(self.img_volumes)
        # print(self.img_subject_vol_index)



    def __getitem__(self, index):

        mask_path = self.mask_volumes[index]
        images_path = self.img_volumes[index]
        # load masks as a volume

        maskname = os.path.basename(mask_path)
        subject_num = maskname.split('_')[0]

        mask = Image.open(mask_path)

        mask = np.array(mask) / 255
        # mask = mask.astype(np.uint8)

        if np.max(mask) >1 or np.min(mask)<0:
            print("ERROR of MASK VALUE : out of range")

        # mask_bg = 1 - mask
        # mask = np.array([mask, mask_bg])
        # mask = torch.as_tensor(mask, dtype=torch.float32)


        image_list=[]
        for image_path in images_path:

            imgname = os.path.basename(image_path)
            if subject_num != imgname.split('_')[0]:
                print("ERROR : Different subject number")
                raise IndexError("unmathced subject")

            image = Image.open(image_path).convert("L")
            # Image._show(image)
            # image = T.ToTensor()(image)
            # image = torch.as_tensor(image, dtype=torch.float32)
            # print(np.unique(image.numpy()))

            # image = torch.unsqueeze(image, 0) # c
            # image = torch.unsqueeze(image, 0) # t

            image = np.array(image)
            image = np.expand_dims(image, 0) #c
            image = np.expand_dims(image, 0) #t

            image_list.extend(image)

        # image_vol = torch.stack(image_list, dim=0)
        image_vol = np.stack(image_list, axis=0)
        # print(image_vol.shape)

        if self.pm_path is not None:
            pm_list = []

            for i in range(len(images_path)):
                # if len(images_path) != self.m_length:
                #     print(images_path)
                image_path = images_path[i]
                imgname = os.path.basename(image_path)
                subject_num = imgname.split('_')[0] + '_1'
                slice_num = imgname.split('_')[-1]
                pm_path = PJ(self.pm_path, subject_num, subject_num + '_' + slice_num + '.npy')
                # print(pm_path)
                pm = np.load(pm_path)
                if np.max(pm) > 1 or np.min(pm) < 0:
                    print("ERROR of PM VALUE : out of range")
                    raise ValueError("PM value out of range")

                # pm = torch.as_tensor(pm, dtype=torch.float32)
                # pm = torch.unsqueeze(pm, 0)  # c
                # pm = torch.unsqueeze(pm, 0)  # t
                # print(pm.shape)

                pm = np.expand_dims(pm, 0)
                pm = np.expand_dims(pm, 0)

                pm_list.extend(pm)

            # pm_vol = torch.stack(pm_list, dim=0)
            pm_vol = np.stack(pm_list,axis=0)
            # print(pm_vol.shape)
            # print(mask.shape)

            if self.transform is not None:
                mask = np.expand_dims(mask, 0)
                mask = np.expand_dims(mask, 0)
                # print(mask.shape, image_vol.shape, pm_vol.shape)
                data = {'image': image_vol, 'pm':pm_vol, 'mask':mask}
                transformed = self.transform(**data)
                image_vol = transformed['image']
                pm_vol = transformed['pm']
                mask = transformed['mask']
                mask = np.squeeze(mask)
            # print(mask.shape)
            image_vol = torch.as_tensor(image_vol, dtype=torch.float32)
            pm_vol = torch.as_tensor(pm_vol, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.float32)


            image_vol = torch.cat((image_vol, pm_vol), dim=1)
            # t, c, w, h
        else:
            if self.transform is not None:
                mask = np.expand_dims(mask, 0)
                mask = np.expand_dims(mask, 0)
                # print(mask.shape, image_vol.shape, pm_vol.shape)
                data = {'image': image_vol, 'mask':mask}
                transformed = self.transform(**data)
                image_vol = transformed['image']
                mask = transformed['mask']
                mask = np.squeeze(mask)
            # print(mask.shape)
            image_vol = torch.as_tensor(image_vol, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.float32)






        return image_vol, mask



    def __len__(self):
        return len(self.img_volumes)


def collate_fn_pad_z(sample):
    i_vol = [s[0] for s in sample]
    m_vol = [s[1] for s in sample]

    i_vol = torch.nn.utils.rnn.pad_sequence(i_vol, batch_first=True)
    m_vol = torch.nn.utils.rnn.pad_sequence(m_vol, batch_first=True)

    return i_vol, m_vol

