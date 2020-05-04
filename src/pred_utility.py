import cv2 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import json
import jsonlines
import os
import glob

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab




def get_target_bbox(bounds_file):
    res={}
    with open(bounds_file, 'r') as f:
        for line in f:
            try:
                para=json.loads(line)
                para['name']=para['name'].lower()
                if para['step'] not in res:
                    res[para['step']]={para['name']:[para]}
                else:
                    if para['name'] not in res[para['step']]:
                        res[para['step']][para['name']]=[para]
                    else:
                        res[para['step']][para['name']].append(para)
            except:
                continue
    return res



def plot_bbox_wrong(img, name, detector_bbox):
    
    txt = '{} p={:.1f}'.format(name, detector_res[4])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    c=(255, 0, 0)
    cv2.rectangle(img, (detector_bbox[0], detector_bbox[1]), (detector_bbox[2], detector_bbox[3]), c, 2)
    cv2.rectangle(img,
                  (detector_bbox[0], detector_bbox[1] - cat_size[1] - 2),
                  (detector_bbox[0] + cat_size[0], detector_bbox[1] - 2), c, -1)
    cv2.putText(img, txt, (detector_bbox[0], detector_bbox[1] - 2), 
                font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    
    
 def IOU(boxA, boxB):
    interAx = max(boxA[0], boxB[0])
    interAy = max(boxA[1], boxB[1])
    interBx = max(boxA[2], boxB[2])
    interBy = max(boxA[3], boxB[3])
    
    interArea = max(0, interBx-interAx+1)*max(0, interBy-interAy+1)
    A_Area    = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    B_Area    = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    return interArea/float(A_Area+B_Area-interArea)

def get_IOU(boxA, boxBs):
    iou=[]
    for boxB in boxBs:
        iou.append(IOU(boxA, boxB))
    return max(iou)




def plot_bbox_right(img, name, detector_bbox, target_info):
    
    target_box = []
    for info in target_info:
        target_box.append([int(info['pixelPosX']), int(info['pixelPosY']), 
                           int(info['pixelPosX']+info['pixelWidth']), int(info['pixelPosY']+info['pixelHeight'])])
    
    IOU = get_IOU(detector_bbox, target_box)
    
    txt = '{} p={:.1f} iou={:.1f}'.format(name, detector_res[4], IOU)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    c=(0, 255, 0)
    cv2.rectangle(img, (detector_bbox[0], detector_bbox[1]), (detector_bbox[2], detector_bbox[3]), c, 2)
    cv2.rectangle(img,
                  (detector_bbox[0], detector_bbox[1] - cat_size[1] - 2),
                  (detector_bbox[0] + cat_size[0], detector_bbox[1] - 2), c, -1)
    cv2.putText(img, txt, (detector_bbox[0], detector_bbox[1] - 2), 
                font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    
    

def Annotate_BoundsInfo(bbox_info, class_names, path):
    
    ctg_dict={}
    for k in range(len(class_names)):
        ctg_dict[class_names[k]]=k
    
    coco_bbox=[]
    for img_id in bbox_info:
        for ctg in bbox_info[img_id]:
            ctg_id = ctg_dict[ctg]
            for box in bbox_info[img_id][ctg]:
                bbox_info = [box['pixelPosX'], box['pixelPosY'], box['pixelWidth'], box['pixelHeight']]
                coco_obj  = {"image_id"    :  img_id,
                             "category_id" :  ctg_id,
                             "bbox"        :  bbox_info,
                             "score"       :  1.0}
                coco_bbox.append(coco_obj)
    
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+'/Coco_target.txt', 'w') as f:
        json.dump(coco_bbox, f, indent=4, sort_keys=True)
    return coco_bbox
