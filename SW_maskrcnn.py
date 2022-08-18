import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import pickle
import cv2
# import transforms as T
# from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import copy
import random

from engine import train_one_epoch
import utils

from SW_SEG_dataset import *
from vol_eval import *

def visualize_augmentations(dataset, idx=30, samples=12, cols=6):
    dataset = copy.deepcopy(dataset)
    dataset.dataset.transform = A.Compose([t for t in dataset.dataset.transform if not isinstance(t, (A.ToFloat, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(0,samples,2):
        image, target = dataset[idx]
        mask = np.array(target['masks'][0].detach().numpy())
        # print(np.unique(mask))
        # print(image.shape)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # print(image.shape)
        # ax.ravel()[i].imshow(cv2.addWeighted(mask,1, image,1,0))
        ax.ravel()[i].imshow(mask)
        ax.ravel()[i+1].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

def get_instance_maskrcnn_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model




def main(mode):
    # random.seed(37)
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    # k-fold
    # start - end range of subject
    # [0, 14] [15, 29] [30, 44] [45, 59] (15, 15, 15, 15)
    test_subject_range = [45, 59]

    weight_path = './SW_SEG_weights/maskrcnn/'
    data_dir = './dataset/FC_pos'
    # data_dir = './dataset/FC_crop_pos'

    train_transform = A.Compose(
        [
            # A.Affine(translate_percent=0.05, rotate=90,interpolation=1), #linear interpolation
            A.Affine(translate_percent=0.05, rotate=(-10,10),interpolation=1), #linear interpolation
            A.ElasticTransform(alpha=1, sigma=10, alpha_affine=10),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    print(weight_path)
    print(test_subject_range)
    #dataset
    dataset = SW_SEG_dataset(data_dir, './dataset/ROI_pos')
    # dataset = SW_SEG_dataset(data_dir, './dataset/ROI_crop_pos')
    # dataset_test = SW_SEG_dataset('AAAHALF_POS', get_transform(train=False))


    if not os.path.isdir(weight_path):
        os.mkdir(weight_path)


    subject_slice = dict()
    split= 0
    # print(dataset.subjects[13], dataset.subjects[25])
    for subject in dataset.subjects:
        # print(subject)
        # print([path for path in dataset.imgs if subject in path])
        num_slice = len([path for path in dataset.imgs if subject in path])
        subject_slice[subject] = [split, split + num_slice]
        split += num_slice
    # subject_slice[subject] indicate start, end slice indices of subject

    # set begining and end of subject number
    total_indicies = list(range(0,len(dataset.imgs)))
    print(len(total_indicies))

    # print(dataset.subjects[15], dataset.subjects[29])

    indices2 = list(range(subject_slice[dataset.subjects[test_subject_range[0]]][0],
                          subject_slice[dataset.subjects[test_subject_range[1]]][1]))
    #ALL train
    # indices2 = []

    indices1 = [index for index in total_indicies if index not in indices2]

    # print(test_subject_range)
    print(len(total_indicies))
    print(len(indices1))
    print(len(indices2))
    print("Total = train + test : %r" %(len(total_indicies)==len(indices1)+len(indices2)))
    # split the dataset in train and test set
    # torch.manual_seed(1)
    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-200])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-200:])

    # # leave subject out
    # indices1 = list(range(0, 9291))
    # indices2 = list(range(9291, 10731))
    # np.random.shuffle(indices1)
    # np.random.shuffle(indices2)


    # dataset split into train / test
    dataset_train = torch.utils.data.Subset(dataset, indices1)
    dataset_train.dataset.transform = train_transform
    # visualize_augmentations(dataset_train)
    dataset_test = torch.utils.data.Subset(dataset, indices2)
    dataset_test.dataset.transform = test_transform


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=5, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn, pin_memory=True)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=1, shuffle=False, num_workers=0,
    #     collate_fn=utils.collate_fn, pin_memory=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and ...
    num_classes = 2

    # get the model using our helper function
    model = get_instance_maskrcnn_model(num_classes)
    # move model to the right device
    model.to(device)

    if 'train' in mode:
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.002,
                                    momentum=0.9, weight_decay=0.0005)
        # optimizer = torch.optim.Adam(params, lr=0.0003, betas=(0.5, 0.999))
        criterion = torch.nn.BCEWithLogitsLoss()

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.5)

        # let's train it for 10 epochs
        num_epochs = 50

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=500)


            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            # evaluate(model, data_loader_test, device=device)

            if epoch % 5 == 0:
                torch.save(model.state_dict(),
                           weight_path + str(test_subject_range[0]) + str(test_subject_range[1]) + '_e'+str(epoch)+'_weight.pth')

        # torch.save(model.state_dict(), './pretrained_SEG/SEG_HALF/4457_pos_weight.pth')
        # torch.save(dataset_test, './pretrained_SEG/SEG_HALF/4457_pos_test.pth')
        torch.save(model.state_dict(), weight_path+str(test_subject_range[0])+str(test_subject_range[1])+'_weight.pth')


    if 'test' in mode:
        # model.load_state_dict(torch.load('./pretrained/pretrained_weight.pth'))
        # data_path = 'AAAHOSEO'
        # data_path = 'dataset'
        weight_path += str(test_subject_range[0])+str(test_subject_range[1])+'_e30_weight.pth'
        print(weight_path)
        model.load_state_dict(torch.load(weight_path))

        threshold, lower, upper = 0.5, 0.0, 1.0
        print(threshold)
        print(lower, upper)

        # img, _ = dataset_test[13]
        # print(img.shape)
        # check_dir = '../AAAGilDatasetPos/'
        # subject = '05390853_20200821'
        #
        # print(subjects)
        # print(len(subjects))
        # subject = '00437393_1'
        # img_idx = 73
        total_ol = []
        total_ja = []
        total_di = []
        total_fp = []
        total_fn = []
        sigmoid = torch.nn.Sigmoid()
        for idx, (image, target) in enumerate(dataset_test):
            image_name = dataset_test.dataset.imgs[indices2[idx]]
            # print(image_name)
            subject = image_name.split('/')[0]
            img_num = image_name.split('/')[1]
            write_path = './result/'+subject+'/'
            if not os.path.isdir(write_path):
                os.mkdir(write_path)

            write_name = os.path.join(write_path,'pred_'+img_num)

            # put the model in evaluation mode
            # print(image.shape)
            # print(type(image))
            if np.isnan(image).any():
                print("imgae have NAN!")
            model.eval()
            with torch.no_grad():
                prediction = model([image.to(device)])

            # im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
            # im.show()
            # print(prediction.shape)
            # print(len(prediction))

            if len(prediction[0]['masks']) > 0:
                #bianry
                pm = prediction[0]['masks'][0,0]
                # prediction = sigmoid(prediction)
                # pm = prediction.squeeze()
                pm = pm.cpu().numpy()
                pm = np.where(pm>threshold, upper, lower)
                # pm[pm > threshold] = upper
                # pm[pm<=threshold] = lower

                # print(np.unique(pm))
                pm *= 255
                pm = pm.astype(np.uint8)
                # print(pm.shape)
                # print(pm)
                # print(pm.dtype)

                mask = Image.fromarray(pm)

                # mask = prediction[0]['masks'][0,0].mul(255).byte().cpu().numpy()
                # mask[mask > 125] = 255
                # mask[mask <= 125] = 0
                # mask = Image.fromarray(mask)

            else:
                mask = np.zeros((512,512))
            # mask.show()
            # mask_gt.show()

            # img_in = np.array(im)
            img_mask = np.array(mask)

            # img_mask_gt = np.array(mask_gt)

            # img_mask[img_mask > 50] = 255
            # img_mask_gt_gray = cv2.cvtColor(img_mask_gt, cv2.COLOR_BGR2GRAY)

            # cv2.imshow('input', img_in)
            # cv2.imshow('mask result', img_mask)
            # cv2.imshow('mask gt', img_mask_gt)

            # segment evaluation
            # overlap, jaccard, dice, fn, fp = eval_volume_from_mask(subject)
            # print('[segmentation evaluation] overlab:%.4f jaccard:%.4f dice:%.4f fn:%.4f fp:%.4f'%(overlap, jaccard, dice, fn, fp))


            # img_result = np.concatenate([img_mask, img_mask_gt_gray], axis=1)

            # img_overlap = img_mask_gt.copy()
            # img_overlap[:, :, 0] = 0
            # img_overlap[:,:,1] = img_mask
            # cv2.imshow('result', img_result)
            # cv2.imshow('overlap', img_overlap)
            # cv2.waitKey(0)
            cv2.imwrite(write_name, img_mask)
        subjects = sorted([name for name in os.listdir('./dataset/ROI_pos/') if name.endswith('_1')])
        subjects = subjects[test_subject_range[0]:test_subject_range[1]+1]

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




if __name__ == '__main__':
    main('train')
