import numpy as np
import os
from PIL import Image

def eval_volume_from_mask(subject, GT_path="./dataset/ROI_pos", pred_path = "./result", unmatched=False):
    # mask file load
    pred_path = os.path.join(pred_path, subject)
    gt_path = os.path.join(GT_path,subject)

    gt_mask_list = sorted([name for name in os.listdir(gt_path) if subject in name])
    start_idx = int(gt_mask_list[0].split('.')[0][-4:])
    finish_idx = int(gt_mask_list[-1].split('.')[0][-4:])
    idx_list = np.arange(start_idx, finish_idx+1)

    pred_mask_list = sorted([name for name in os.listdir(pred_path) if subject in name])
    pred_mask_filtering = []
    for name in pred_mask_list:
        if int(name.split('.')[0][-4:]) in idx_list:
            pred_mask_filtering.append(name)

    pred_mask_list = pred_mask_filtering


    # calculation
    s_sum, t_sum = 0, 0
    intersection, union = 0, 0
    s_diff_t, t_diff_s = 0, 0


    if(len(gt_mask_list) == len(pred_mask_list)) and not unmatched:
        for i in range(len(gt_mask_list)):
            # name matched
            if gt_mask_list[i].split('_')[-1] == pred_mask_list[i].split('_')[-1]:


                gt_slice = Image.open(os.path.join(gt_path, gt_mask_list[i])).convert("RGB")
                gt_slice = (np.array(gt_slice)/255.0).astype(np.uint32)
                pred_slice = Image.open(os.path.join(pred_path, pred_mask_list[i])).convert("RGB")
                pred_slice = (np.array(pred_slice) / 255.0).astype(np.uint32)

                # print(pred_mask_list[i], gt_mask_list[i])
                # print(gt_slice.shape, pred_slice.shape)

                if len(np.unique(gt_slice)) > 2:
                    print("GT SLICE VALUE ERROR! NOT 0 ,1")
                    print(np.unique(gt_slice))
                if len(np.unique(pred_slice)) > 2:
                    print("PRED SLICE VALUE ERROR! NOT 0 ,1")
                    print(np.unique(pred_slice))
                # print(subject)
                # print(pred_slice.shape, gt_slice.shape)
                s_sum += (pred_slice == 1).sum()
                t_sum += (gt_slice == 1).sum()

                intersection += np.bitwise_and(pred_slice, gt_slice).sum()
                union += np.bitwise_or(pred_slice, gt_slice).sum()

                s_diff_t += (pred_slice - np.bitwise_and(pred_slice, gt_slice)).sum()
                # print(np.unique(np.bitwise_and(pred_slice, gt_slice)))
                # print(np.unique(pred_slice - np.bitwise_and(pred_slice, gt_slice)))
                t_diff_s += (gt_slice - np.bitwise_and(pred_slice, gt_slice)).sum()

                # print(gt_mask_list[i].split('_')[-1], pred_mask_list[i].split('_')[-1])
            # print(t_sum, s_sum)
            # print(intersection)
            # print(s_diff_t)

        overlab = intersection / t_sum
        jaccard = intersection / union
        dice = 2.0*intersection / (s_sum + t_sum)
        fn = t_diff_s / t_sum
        fp = s_diff_t / s_sum

        return overlab, jaccard, dice, fn, fp
    elif (len(gt_mask_list) >= len(pred_mask_list)) and unmatched:
        # print(subject)
        pred_index = 0
        for i in range(len(gt_mask_list)):
            # name matched
            # print(i, pred_index)
            # print(len(gt_mask_list), len(pred_mask_list))
            if pred_index < len(pred_mask_list) and gt_mask_list[i].split('_')[-1] == pred_mask_list[pred_index].split('_')[-1]:

                gt_slice = Image.open(os.path.join(gt_path, gt_mask_list[i])).convert("RGB")
                gt_slice = (np.array(gt_slice)/255.0).astype(np.uint32)
                pred_slice = Image.open(os.path.join(pred_path, pred_mask_list[pred_index])).convert("RGB")
                pred_slice = (np.array(pred_slice) / 255.0).astype(np.uint32)
                # print(pred_mask_list[pred_index], gt_mask_list[i])
                pred_index += 1
            else:
                gt_slice = Image.open(os.path.join(gt_path, gt_mask_list[i])).convert("RGB")
                gt_slice = (np.array(gt_slice)/255.0).astype(np.uint32)
                pred_slice = np.zeros((256,256,3), dtype=np.uint32)
                # print(gt_mask_list[i])


            # print(gt_slice.shape, pred_slice.shape)

            if len(np.unique(gt_slice)) > 2:
                print("GT SLICE VALUE ERROR! NOT 0 ,1")
                print(np.unique(gt_slice))
            if len(np.unique(pred_slice)) > 2:
                print("PRED SLICE VALUE ERROR! NOT 0 ,1")
                print(np.unique(pred_slice))
            # print(subject)
            # print(pred_slice.shape, gt_slice.shape)
            s_sum += (pred_slice == 1).sum()
            t_sum += (gt_slice == 1).sum()

            intersection += np.bitwise_and(pred_slice, gt_slice).sum()
            union += np.bitwise_or(pred_slice, gt_slice).sum()

            s_diff_t += (pred_slice - np.bitwise_and(pred_slice, gt_slice)).sum()
            # print(np.unique(np.bitwise_and(pred_slice, gt_slice)))
            # print(np.unique(pred_slice - np.bitwise_and(pred_slice, gt_slice)))
            t_diff_s += (gt_slice - np.bitwise_and(pred_slice, gt_slice)).sum()

            # print(gt_mask_list[i].split('_')[-1], pred_mask_list[i].split('_')[-1])
            # print(t_sum, s_sum)
            # print(intersection)
            # print(s_diff_t)

        if t_sum != 0:
            overlab = intersection / t_sum
            jaccard = intersection / union
            dice = 2.0*intersection / (s_sum + t_sum)
            fn = t_diff_s / t_sum
            fp = s_diff_t / s_sum

            return overlab, jaccard, dice, fn, fp
        else:
            return 0, 0, 0, 0, 0
    else:
        print("GT : %d, Pred : %d" %(len(gt_mask_list), len(pred_mask_list)))
        print(subject + " ERROR")




