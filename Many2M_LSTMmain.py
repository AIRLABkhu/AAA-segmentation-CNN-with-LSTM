import torchvision
from torchsummary import summary
# from IMDataset import *
from IMvolDataset import *

import volumentations as volaug

from pm_generator.maskrcnn_pm_generator import maskrcnn_pm_generator
from pm_generator.unet_pm_generator import unet_pm_generator
from pm_generator.mia_pm_generator import mia_pm_generator
from pm_generator.uPP_pm_generator import uPP_pm_generator

from models.convLSTM import *
from models.straightNet import *
from models.block_clstm import multi_block_LSTM
# from models.senosr3D import Senosr3D

from models.unet_model import UNet
from result_compare import *


import cv2

from torchvision.transforms import functional as F

from vol_eval import *
import os
import logging



def load_checkPoint(PATH, model, optimizer=None, scheduler=None, epoch=None, loss=None, only_model=True):
    checkpoint= torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    if not only_model:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return {'model':model, 'optimizer':optimizer, 'scheduler':scheduler, 'epoch':epoch, 'loss':loss}
    else:
        return model

def save_checkPoint(PATH, model, optimizer, scheduler, epoch, loss):
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),
        'loss':loss,
    },PATH)


def main(mode, gpu, exp_prefix='test', train_epoch=100,save_epoch=None, load_epoch=None, resume_exp=None, slice=1, batch_size=1):


    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("DEVICE :",device, gpu , "PREFIX :",exp_prefix)

    bhw = (batch_size,512,512)

    # summary(model, (slice,2,256,256),batch_size=batch_size)


    ## M2M
    data_path = './dataset/FC_pos'

    pm_path = './pred_map' + str(gpu)
    # pm_path = None

    save_path = os.path.join("./3Dexperiment/",exp_prefix)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    threshold = 0.5

    train_transform = volaug.Compose([
        volaug.RotatePseudo2D(axes=(2,3), limit=(-10,10), interpolation=1),
        volaug.ElasticTransformPseudo2D(alpha=1, sigma=10, alpha_affine=10),
        volaug.Normalize(),
    ])

    test_transform = volaug.Compose([
        volaug.Normalize(),
    ])

    # [0, 14] [15, 29] [30, 44] [45, 59] (15, 15, 15, 15)
    test_index_list = [(0, 14), (15, 29), (30, 44), (45, 59)]
    logger = logging.getLogger("tracker")
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    log_file = logging.FileHandler(save_path+'/train.log')
    log_file.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(log_file)

    ### test index for resume experiment
    # resume_exp = ([test_index],[start_epoch])
    if resume_exp is not None and mode == 'train':
        test_index_list = [test_index_list[i] for i in resume_exp[0]]

    if 'train' in mode:
        for i, test_index in enumerate(test_index_list):


            ## used this

            model = straightNetBack(in_channels=2, conv_channel=64, lstm_channel=64, HALF_MODE=False)

            # model = BiConvLSTM(input_channel=2, num_filter=64, output_channel=1, b_h_w=bhw, kernel_size=(3,3))

            model.to(device)

            print('==============================')
            print(test_index)

            subjects = sorted([name for name in os.listdir('./dataset/ROI_pos') if name.endswith('_1')])
            # select test subject range
            test_subjects = subjects[test_index[0]:test_index[1]+1]
            subjects = [index for index in subjects if index not in test_subjects]

            print(subjects)
            print('train subject Num ', len(subjects))
            print('test subject NUm ', len(test_subjects))
            ### maskrcnn
            maskrcnn_pm_generator('%d%d_e30_weight.pth' %(test_index[0],test_index[1]),
                                  model_path='./SW_SEG_weights/maskrcnn',
                                  dir_path=pm_path, data_path=data_path, subjects=test_index)
            ### unet
            # unet_pm_generator('%d%d_e40_weight.pth' %(test_index[0],test_index[1]),
            #                       model_path='./SW_SEG_weights/unetAdam',
            #                       dir_path=pm_path, data_path=data_path, subjects=test_index)
            ### mia
            # mia_pm_generator('%d%d_e35_weight.pth' %(test_index[0],test_index[1]),
            #                       model_path='./SW_SEG_weights/mia',
            #                       dir_path=pm_path, data_path=data_path, subjects=test_index)
            ### unet++
            # uPP_pm_generator('%d%d_e30_weight.pth' %(test_index[0],test_index[1]),
            #                       model_path='./SW_SEG_weights/unetPP',
            #                       dir_path=pm_path, data_path=data_path, subjects=test_index)



            print("==PM generataion is done==")
            # dataset declare
            # M2M
            dataset = IMDataset(data_path, max_slice=slice, pm_path=pm_path)
            # dataset = IMDataset(data_path, max_slice=slice, pm_path=pm_path, only_pm=True)



            # indicies
            train_indicies = []

            for subject in subjects:
                train_indicies.extend(dataset.img_subject_vol_index[subject])

            test_indicies = []
            for subject in test_subjects:
                test_indicies.extend(dataset.img_subject_vol_index[subject])



            print(len(dataset), len(test_indicies), len(train_indicies))

            train_dataset = torch.utils.data.Subset(dataset, train_indicies)
            train_dataset.dataset.transform = train_transform
            # define dataloader
            data_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn_pad_z, pin_memory=True)

            # construct an optimizer
            params = [p for p in model.parameters() if p.requires_grad]

            criterion = torch.nn.BCEWithLogitsLoss()


            optimizer = torch.optim.SGD(params, lr=0.005,
                                        momentum=0.9, weight_decay=0.0005)

            # 2
            # optimizer = torch.optim.SGD(params, lr=0.002,
            #                             momentum=0.9, weight_decay=0.0005)

            # optimizer = torch.optim.Adam(params, lr=0.0001, betas=(0.5, 0.999))

            # and a learning rate scheduler which decreases the learning rate by
            # 10x every 3 epochs
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                           step_size=3,
                                                           gamma=0.1)

            start_epoch=0
            ### load experiment
            if resume_exp is not None:
                start_epoch = resume_exp[1][i]
                if start_epoch > 0:
                    checkpoint = load_checkPoint(PATH=os.path.join(save_path, '%d%d_pos_LSTM_weight_e%d.pth' % (test_index[0], test_index[1], start_epoch),),
                                                 model=model, optimizer=optimizer, scheduler=lr_scheduler, only_model=False)
                    model = checkpoint['model'].to(device)
                    optimizer = checkpoint['optimizer']
                    lr_scheduler = checkpoint['scheduler']
                    running_loss = checkpoint['loss']
                    print('Load epcoh : %d, start epoch : %d, === Load Loss %f' %(checkpoint['epoch'], start_epoch, running_loss))
                    start_epoch = checkpoint['epoch']




            for epoch in range(start_epoch+1, train_epoch+1):
                running_loss = 0
                logger.info("===EPOCH %d ===" %epoch)
                msg=True
                for i, (volume, target) in enumerate(data_loader):
                    model.train()
                    optimizer.zero_grad()

                    volume=volume.to(device)
                    target=target.to(device)

                    pred, _ = model(volume, device=device)
                    #unet
                    # pred = model(volume)
                    # sensor3D
                    # pred = model(volume, device=device)

                    if msg:
                        print("vol shape ", volume.shape)
                        print("pred shape ", pred.shape)
                        print("Target shape ",target.shape)
                        msg=False

                    pred = pred.squeeze()
                    target = target.squeeze()


                    loss = criterion(pred, target)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    del volume, target

                    if i % 20 == 19:
                        logger.info("[%d %d] loss : %f" %(epoch, i, running_loss/20))
                        running_loss = 0


                lr_scheduler.step()
                torch.cuda.empty_cache()

                if epoch in save_epoch:
                    # torch.save(model.state_dict(),
                    #            os.path.join(save_path, '%d%d_pos_LSTM_weight_e%d.pth' % (test_index[0], test_index[1], epoch)))
                    save_checkPoint(PATH=os.path.join(save_path, '%d%d_pos_LSTM_weight_e%d.pth' % (test_index[0], test_index[1], epoch)),
                                    model=model, optimizer=optimizer, scheduler=lr_scheduler, epoch=epoch, loss=running_loss)
                    logger.info("==SAVE MODEL : %d%d_pos_LSTM_weight_e%d.pth==" % (test_index[0], test_index[1], epoch))

            save_checkPoint(
                PATH=os.path.join(save_path, '%d%d_pos_LSTM_weight_e%d.pth' % (test_index[0], test_index[1], epoch)),
                model=model, optimizer=optimizer, scheduler=lr_scheduler, epoch=epoch, loss=running_loss)
            logger.info("==SAVE MODEL : %d%d_pos_LSTM_weight_e%d.pth==" %(test_index[0], test_index[1], epoch))


    elif 'eval' in mode:

        if load_epoch is None:
            load_epoch = train_epoch

        #########
        total_ol = []
        total_ja = []
        total_di = []
        total_fp = []
        total_fn = []

        for test_index in test_index_list:

            model = straightNetBack(in_channels=2, conv_channel=64, lstm_channel=64)
            # model = BiConvLSTM(input_channel=2, num_filter=64, output_channel=1, b_h_w=bhw, kernel_size=(3,3))


            model.to(device)

            subjects = sorted([name for name in os.listdir('./dataset/ROI_pos') if name.endswith('_1')])
            # select test subject range
            subjects = subjects[test_index[0]:test_index[1] + 1]
            # print("TEST SUTJCET LIST : ", subjects)


            model_path = "%d%d_pos_LSTM_weight_e%d.pth" %(test_index[0], test_index[1], load_epoch)


            ### maskrcnn
            maskrcnn_pm_generator('%d%d_e30_weight.pth' %(test_index[0],test_index[1]),
                                  model_path='./SW_SEG_weights/maskrcnn',
                                  dir_path=pm_path, data_path=data_path, subjects=test_index, test_mode=True)
            ### unet
            # unet_pm_generator('%d%d_e40_weight.pth' %(test_index[0],test_index[1]),
            #                       model_path='./SW_SEG_weights/unetAdam',
            #                       dir_path=pm_path, data_path=data_path, subjects=test_index, test_mode=True)
            ### mia
            # mia_pm_generator('%d%d_e35_weight.pth' %(test_index[0],test_index[1]),
            #                       model_path='./SW_SEG_weights/mia',
            #                       dir_path=pm_path, data_path=data_path, subjects=test_index, test_mode=True)
            ### unet++
            # uPP_pm_generator('%d%d_e30_weight.pth' %(test_index[0],test_index[1]),
            #                       model_path='./SW_SEG_weights/unetPP',
            #                       dir_path=pm_path, data_path=data_path, subjects=test_index, test_mode=True)

            model_path = os.path.join(save_path,model_path)
            model=load_checkPoint(model_path, model)
            # model.load_state_dict(torch.load(model_path))
            model.to(device)



            # dataset declare
            data_path = './dataset/FC_pos'
            dataset = IMDataset(data_path, max_slice=slice, pm_path=pm_path)
            # dataset = IMDataset(data_path, max_slice=slice, pm_path=pm_path, only_pm=True)
            dataset.transform = test_transform

            # indices
            test_indicies = []
            # print(subjects)
            for subject in subjects:
                test_indicies.extend(dataset.img_subject_vol_index[subject])

            # print("test idice/s : ",test_indicies)


            sigmoid = torch.nn.Sigmoid()
            # total_data = np.array([])
            for idx in test_indicies:
                model.eval()
                with torch.no_grad():
                    volume, target = dataset[idx]
                    # print(volume.shape)
                    # print(target.shape)
                    volume=volume.unsqueeze(0)
                    # print(volume.shape)

                    volume = volume.to(device)
                    # target = target.to(device)


                    prediction, _ = model(volume, device=device)

                    # unet
                    # prediction = model(volume)

                    # prediction = prediction[-1]
                    # print(prediction.shape)
                prediction = prediction.squeeze(0) #b
                # print(prediction.shape)
                # prediction = softmax(prediction)
                # print(prediction[0][0] + prediction[0][1])
                # print("=============================" * 2)nvidia
                prediction = sigmoid(prediction)
                ## result name
                # print(prediction.shape)
                # print(len(prediction), len(dataset.mask_volumes[idx]))

                for i in range(len(prediction)):
                    # print(len(dataset.img_volumes[idx]))
                    # print(dataset.mask_volumes[idx])
                    # print(dataset.img_volumes[idx])
                    # print("==============")

                    path = dataset.mask_volumes[idx][i]
                    # print(len(prediction), len(dataset.mask_volumes[idx]))
                    # path = dataset.img_volumes[idx][i]
                    # print(path)

                    write_path = './result_lstm/' + path.split('/')[3]+'/'

                    if not os.path.isdir(write_path):
                        os.mkdir(write_path)

                    write_name = os.path.join(write_path, 'pred_'+path.split('/')[-1])
                    # print(write_name)

                    # print(prediction[i][0])
                    # mask = prediction[i][0].mul(255).byte().cpu().numpy()
                    # print("Pred max : %f // min : %f" %(torch.max(prediction[i]), torch.min(prediction[i])))

                    mask = prediction[i].cpu().numpy()  # BCE loss
                    # print("MASK max : %f // min : %f" % (np.max(mask), np.min(mask)))
                    #binary
                    mask = np.where(mask>threshold, 1.0, 0.0)
                    mask *= 255
                    mask = mask.astype(np.uint8)
                    # print("MASK max : %f // min : %f" % (np.max(mask), np.min(mask)))
                    # print(mask.shape)
                    # this
                    mask = mask.squeeze(0) # c
                    # print(mask.shape)
                    mask = Image.fromarray(mask)
                    # mask.show()
                    img_mask = np.array(mask)
                    # data = img_mask.flatten()
                    ## for histogram
                    # total_data = np.concatenate((total_data, data), axis=0)
                    # print(data)
                    # print(data.ndim)

                    # img_mask[img_mask>70] = 255
                    # img_mask[img_mask<=70] = 0
                    # cv2.imshow('mask result', img_mask)
                    cv2.imwrite(write_name, img_mask)
                    # _ = plt.hist(data, bins='auto')
                    # plt.show()
                    # plt.close('all')
                    # cv2.waitKey(0)


            for s in subjects:
                overlap, jaccard, dice, fn, fp = eval_volume_from_mask(s, pred_path = "./result_lstm")
                print(s +' overlap: %.4f jaccard: %.4f dice: %.4f fn: %.4f fp: %.4f' % (
                overlap, jaccard, dice, fn, fp))
                total_ol.append(overlap)
                total_ja.append(jaccard)
                total_di.append(dice)
                total_fn.append(fn)
                total_fp.append(fp)
                # print(subject, overlap, jaccard, dice, fn, fp)
            # for s in subjects:
            #     result_compare('result_lstm','AAAHALF_POS/ROI',s, 'compare')
        print('Average overlap: %.4f jaccard: %.4f dice: %.4f fn: %.4f fp: %.4f' % (
            np.mean(total_ol), np.mean(total_ja), np.mean(total_di), np.mean(total_fn), np.mean(total_fp)))

            # histogram total data
            # print(total_data.shape)
            # print("MAX:  %d, MIN: %d" %(np.max(total_data), np.min(total_data)))
            # _ = plt.hist(total_data, bins='auto', density=True)
            # plt.show()
            # plt.close('all')

if __name__ == '__main__':


    resume_exp=None


    ### 3 5 7 9 M2M

    # main('train', exp_prefix='pi_biconv_64_64_s3b2', gpu=1, train_epoch=5, save_epoch=[5,10,20,30,50,100,150],
    #      load_epoch=5, slice=3, batch_size=2, resume_exp=resume_exp)
    # main('eval', exp_prefix='img_biconv_64_64_s3b2', gpu=1, train_epoch=5, save_epoch=[5,10,20,30,50,100,150],
    #      load_epoch=5, slice=3, batch_size=2, resume_exp=resume_exp)


    # main('eval', exp_prefix='pi_snBack_64_64_s3b2', gpu=1, train_epoch=30, save_epoch=[5,10,20,30,50,100,150],
    #      load_epoch=30, slice=3, batch_size=2, resume_exp=resume_exp)
    # main('eval', exp_prefix='pi_snBack_64_64_s5b2', gpu=1, train_epoch=30, save_epoch=[5,10,20,30,50,100,150],
    #      load_epoch=30, slice=5, batch_size=2, resume_exp=resume_exp)
    main('eval', exp_prefix='pi_snBack_64_64_s7b1', gpu=1, train_epoch=30, save_epoch=[5,10,20,30,50,100,150],
         load_epoch=20, slice=7, batch_size=1, resume_exp=resume_exp)



    '''
    ## trash
    # main('eval', exp_prefix='sn_Mid_32_32_half' ,gpu=3 ,train_epoch=200, save_epoch=[30,50,100,150],HALF_MODE=True, load_epoch=200)
    # main('eval', exp_prefix='sn_Frt_32_32_half' ,gpu=3 ,train_epoch=200, save_epoch=[30,50,100,150],HALF_MODE=True, load_epoch=30)
    '''

