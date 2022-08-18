import os
from os.path import join as PJ
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as T

class IMDataset(torch.utils.data.Dataset):
    # max_slice = None => Whole subject slice as a volume

    def __init__(self, data_path, max_slice=None, pm_path = None, only_pm = False, transform=None):

        self.data_path = data_path
        self.max_slice = max_slice
        self.pm_path = pm_path
        self.only_pm = only_pm
        self.transform = transform

        #### hoseo_AAA

        img_subjects = [name for name in os.listdir(os.path.join(data_path))
                         if os.path.isdir(os.path.join(data_path,name))]

        self.subjects = sorted(img_subjects)

        self.img_subject_vol_index = {}
        self.img_volumes = []
        img_index = 0
        ## subject into volume
        for subject in img_subjects:
            images_path = [PJ(data_path, subject, img_path)
                          for img_path in os.listdir(PJ(data_path, subject))
                          if img_path.endswith('png')]
            images_path = sorted(images_path)

            #volume as whole subject
            if max_slice is None:
                self.img_subject_vol_index[subject]=[img_index]
                self.img_volumes.append(images_path)
                img_index += 1
            #volume as sub volume of subject
            else:
                idx_list = []
                # for i in range(0, len(images_path)-max_slice+1):
                for i in range(0, len(images_path),max_slice):
                    # print(len(images_path), i, i+max_slice, -max_slice - 1)
                    if i+max_slice < len(images_path)+1:
                        self.img_volumes.append(images_path[i:i+max_slice])
                        # print(len(images_path[i:i+max_slice]))
                    else:
                        self.img_volumes.append(images_path[-max_slice :])
                        # print(len(images_path[-max_slice-1 : -1]))
                    idx_list.append(img_index)
                    img_index += 1

                self.img_subject_vol_index[subject]=idx_list


        self.mask_subject_vol_index = {}
        self.mask_volumes = []
        mask_index = 0
        ## subject into volume
        for subject in img_subjects:

            masks_path = [PJ("./dataset", "ROI_pos", subject, img_path)
                          for img_path in os.listdir(PJ('./dataset/ROI_pos', subject))
                          if img_path.endswith('png')]
            masks_path = sorted(masks_path)

            #volume as whole subject
            if max_slice is None:
                self.mask_subject_vol_index[subject]=[mask_index]
                self.mask_volumes.append(masks_path)
                mask_index += 1
            #volume as sub volume of subject
            else:
                idx_list = []
                # for i in range(0, len(masks_path)-max_slice+1):
                for i in range(0, len(masks_path), max_slice):
                    if i + max_slice < len(masks_path) + 1:
                        self.mask_volumes.append(masks_path[i:i+max_slice])
                    else:
                        self.mask_volumes.append(masks_path[-max_slice:])
                        # self.mask_volumes.append(masks_path[i:-1])
                    idx_list.append(mask_index)
                    mask_index += 1

                self.mask_subject_vol_index[subject]=idx_list

        # print(len(self.img_volumes))
        # print(self.mask_subject_vol_index)
        if self.mask_subject_vol_index != self.img_subject_vol_index:
            print(self.mask_subject_vol_index)
            print(self.img_subject_vol_index)
            print("IMAGES AND MASKS ARE NOT MATCHED!")
            raise Exception("unmatched image and mask set")

        # self.pm_volumes = []
        # if self.pm_path is not None:
        #     for subject in img_subjects:
        #         pms_path = [PJ(pm_path, subject, pm)
        #                     for pm in os.listdir(PJ(pm_path, subject))
        #                     if pm.endswith('.npy')]
        #         pms_path = sorted(pms_path)
        #
        #         if max_slice is None:
        #             self.pm_volumes.append(pms_path)
        #         else:
        #             for  i in range(0, len(pms_path),max_slice):
        #                 try:
        #                     self.pm_volumes.append(pms_path[i:i+max_slice])
        #                 except IndexError:
        #                     self.pm_volumes.append(pms_path[i:-1])





    def __getitem__(self, index):

        masks_path = self.mask_volumes[index]
        images_path = self.img_volumes[index]

        image_list=[]
        mask_list=[]

        ## check data length
        if not len(masks_path) == len(images_path):
            print("ERROR : length is unmatched!")
            raise IndexError("data Length is unmatched")

        for i in range(len(masks_path)):

            mask_path = masks_path[i]
            image_path = images_path[i]
            maskname = os.path.basename(mask_path)
            imgname = os.path.basename(image_path)

            if (maskname.split('_')[0] != imgname.split('_')[0]):
                print(maskname.split('_')[0], imgname.split('_')[0])
                print("ERROR : Different subject number")
                raise IndexError("unmathced subject")
            # print(maskname.split('_')[-1], imgname.split('_')[-1])
            if (maskname.split('_')[-1] != imgname.split('_')[-1]) :
                print("ERROR : Different slice number")
                raise IndexError("unmatched slice")

            #### mask
            mask = Image.open(mask_path)
            # mask = Image.open(mask_path).convert("L")

            mask = np.array(mask) / 255
            # mask = mask.astype(np.uint8)

            if np.max(mask) >1 or np.min(mask)<0:
                print("ERROR of MASK VALUE : out of range")
                raise ValueError("Mask value out of range")

            # mask_bg = 1 - mask
            # mask = np.array([mask, mask_bg])
            # mask = torch.as_tensor(mask, dtype=torch.float32)
            # mask = torch.unsqueeze(mask, 0) # c
            # mask = torch.unsqueeze(mask, 0) # t
            # mask_list.extend(mask)
            mask = np.expand_dims(mask, 0)
            mask = np.expand_dims(mask, 0)
            # print("m", mask.shape)
            mask_list.extend(mask)

            #### image
            image = Image.open(image_path).convert("L")
            # image = Image.open(image_path)
            # Image._show(image)
            # image = T.ToTensor()(image)
            # image = image.cpu().detach().numpy()
            # print(np.max(image), np.min(image))
            ## image = torch.as_tensor(image, dtype=torch.float32)
            ## print(np.unique(image.numpy()))

            ## image = torch.unsqueeze(image, 0) # c
            # image = torch.unsqueeze(image, 0)  # t
            # image_list.extend(image)

            # image = np.array(image)
            image = np.array(image) / 255
            image = np.expand_dims(image, 0) #c
            image = np.expand_dims(image, 0) #t
            # print("i", image.shape)
            image_list.extend(image)

        # image_vol = torch.stack(image_list, dim=0)
        # mask_vol = torch.stack(mask_list, dim=0)

        image_vol = np.stack(image_list, axis=0)
        mask_vol = np.stack(mask_list, axis=0)
        # print("iv", image_vol.shape)
        # print("mv", mask_vol.shape)

        if self.pm_path is not None:
            pm_list = []

            for i in range(len(images_path)):
                image_path = images_path[i]
                imgname = os.path.basename(image_path)
                subject_num = imgname.split('_')[0]+'_1'
                slice_num = imgname.split('_')[-1]
                pm_path = PJ(self.pm_path, subject_num, subject_num+'_'+slice_num+'.npy')
                # print(pm_path)
                pm = np.load(pm_path)
                # print(np.unique(pm))

                if np.max(pm) > 1 or np.min(pm) < 0:
                    print("ERROR of PM VALUE : out of range")
                    raise ValueError("PM value out of range")

                # Image._show(Image.fromarray(pm*255))
                # print(np.unique(pm))
                # pm = torch.as_tensor(pm, dtype=torch.float32)
                # pm = torch.unsqueeze(pm, 0) #c
                # pm = torch.unsqueeze(pm, 0) #t
                # pm_list.extend(pm)
                pm = np.expand_dims(pm, 0)
                pm = np.expand_dims(pm, 0)

                pm_list.extend(pm)

            # pm_vol = torch.stack(pm_list, dim=0)
            pm_vol = np.stack(pm_list,axis=0)
            # print(image_vol.shape)
            # print(pm_vol.shape)
            if self.only_pm:
                if self.transform is not None:
                    # mask = np.expand_dims(mask, 0)
                    # mask = np.expand_dims(mask, 0)
                    # print(mask.shape, image_vol.shape, pm_vol.shape)
                    print("################transform2")
                    data = {'image': pm_vol, 'mask': mask_vol}
                    transformed = self.transform(**data)
                    pm_vol = transformed['image']
                    mask_vol = transformed['mask']

                pm_vol = torch.as_tensor(pm_vol, dtype=torch.float32)
                mask_vol = torch.as_tensor(mask_vol, dtype=torch.float32)
                return pm_vol, mask_vol
            else:
                # image_vol = torch.cat((image_vol,pm_vol), dim=1)
                image_vol = np.concatenate([image_vol,pm_vol], axis=1)
                # print(image_vol.shape)
        #
        # if self.transform is not None:
        #     # mask = np.expand_dims(mask, 0)
        #     # mask = np.expand_dims(mask, 0)
        #     # print(mask.shape, image_vol.shape, pm_vol.shape)
        #     print("################transform1")
        #     data = {'image': image_vol, 'mask':mask_vol}
        #     transformed = self.transform(**data)
        #     image_vol = transformed['image']
        #     mask_vol = transformed['mask']

        image_vol = torch.as_tensor(image_vol, dtype=torch.float32)
        mask_vol = torch.as_tensor(mask_vol, dtype=torch.float32)
        # print(torch.max(image_vol), torch.min(image_vol))
        # print("iv", image_vol.shape)
        # print("mv", mask_vol.shape)
        return image_vol, mask_vol



    def __len__(self):
        return len(self.img_volumes)


def collate_fn_pad_z(sample):

    i_vol = [s[0] for s in sample]
    m_vol = [s[1] for s in sample]

    i_vol = torch.nn.utils.rnn.pad_sequence(i_vol, batch_first=True)
    m_vol = torch.nn.utils.rnn.pad_sequence(m_vol, batch_first=True)

    return i_vol, m_vol

