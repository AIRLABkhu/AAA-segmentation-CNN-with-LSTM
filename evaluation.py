import os
from vol_eval import *
import cv2

if __name__ =="__main__":

    result_path = './result/result_not_unetpp'

    total_ol = []
    total_ja = []
    total_di = []
    total_fn = []
    total_fp = []

    subjects = sorted([name for name in os.listdir('./dataset/ROI_pos') if name.endswith('_1')])

    for s in subjects:
        overlap, jaccard, dice, fn, fp = eval_volume_from_mask(s, pred_path=result_path)
        print(s + ' overlap: %.4f dice: %.4f jaccard: %.4f  fn: %.4f fp: %.4f' % (
            overlap, dice, jaccard, fn, fp))
        total_ol.append(overlap)
        total_ja.append(jaccard)
        total_di.append(dice)
        total_fn.append(fn)
        total_fp.append(fp)

    print('Average overlap: %.4f dice: %.4f jaccard: %.4f fn: %.4f fp: %.4f' % (
        np.mean(total_ol), np.mean(total_di), np.mean(total_ja), np.mean(total_fn), np.mean(total_fp)))

    #
    # path = './pred_map_unetpp/'
    # dst_path = './result/result_not_unetpp'
    #
    # for sub in os.listdir(path):
    #     print("%s"%sub)
    #     if not os.path.isdir(os.path.join(dst_path, sub)):
    #         os.mkdir(os.path.join(dst_path, sub))
    #
    #     for sub_idx in os.listdir(os.path.join(path, sub)):
    #         pred = np.load(os.path.join(path,sub,sub_idx))
    #         pred *= 255
    #         pred = pred.astype(np.uint8)
    #         pred[pred > 127] = 255
    #         pred[pred <= 127] = 0
    #
    #         cv2.imwrite(os.path.join(dst_path, sub, sub_idx.split('.')[0]+'.png'), pred)

