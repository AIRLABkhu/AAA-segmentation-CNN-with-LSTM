import numpy as np
import os
import cv2
from PIL import Image


def result_compare(pred_path, GT_path, subject, save_path):
      if not os.path.isdir(save_path):
          os.mkdir(save_path)

      if not os.path.isdir(os.path.join(save_path, subject)):
          os.mkdir(os.path.join(save_path,subject))

      pred_list_path = os.path.join(pred_path, subject)
      gt_list_path = os.path.join(GT_path, subject)

      pred_list = [name for name in os.listdir(pred_list_path) if name.endswith('png')]
      gt_list = [name for name in os.listdir(gt_list_path) if name.endswith('png')]

      pred_list = sorted(pred_list)
      gt_list = sorted(gt_list)

      for i, pred in enumerate(pred_list):
          p_image = os.path.join(pred_list_path, pred)
          g_image = os.path.join(gt_list_path,gt_list[i])

          p = Image.open(p_image).convert("RGB")
          g = Image.open(g_image)

          p = np.array(p)
          g = np.array(g)
          # print(p.shape, g.shape)

          p[:,:,0]=0
          p[:,:,1]=g
          cv2.putText(p, 'y : overlap, r : fp, g : fn', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1 , (255,255,255))
          cv2.imwrite(os.path.join(save_path,subject,pred),p)


if __name__ == "__main__":
    subjects = sorted([name for name in os.listdir('AAAHALF_POS' + '/ROI/') if name.endswith('_1')])
    for s in subjects:
        result_compare('./result_lstm_pi_snb_64_64/','AAAHOSEO_POS/ROI',s, 'compare')