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
from Pred_utility import *
import argparse


import sys
CENTERNET_PATH = "./lib/"
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts




def Gaila_Detection(Gaila_path, Task_ids, Scene_ids, MODEL_PATH, class_names,  SampleRate=1, Vis_thresh=0.3):
    
    Output_path= './Output/'
    TASK = 'ctdet'
    opt  = opts().init('{} --load_model {} --vis_thresh {}'.format(TASK, MODEL_PATH, Vis_thresh).split(' '))
    detector = detector_factory[opt.task](opt)


    for scene in Scene_ids:
        for task in Task_ids:
            bbox_path  = Gaila_path+'/bounds/'+str(scene)+'_task'+str(task)+'_bounds.txt'
            bbox_info  = get_target_bbox(bbox_path)
            save_path  = Output_path+'/DectectionResults/'+str(scene)+'/'+str(scene)+'_task'+str(task)+'/'
            coco_target= Annotate_BoundsInfo(bbox_info, class_names, save_path)
            if not os.path.exists(out_path):
                os.makedirs(out_path)


            image_path = Gaila_path+'/images_10hz/'+str(scene)+'/'+str(scene)+'_task'+str(task)+'/'
            image_files= glob.glob(image_path+'/*.png')
            coco_det   = []
            for k in range(0, len(image_files), SampleRate):
                image    = image_files[k]
                step     = int(image[-9:-5])
                det_res  = detector.run(image)['results']
                img      = mpimg.imread(image) 

                for catID, obj in det_res.items():
                    if class_names[catID-1] not in bbox_info[step]:
                        for res in obj:
                            plot_bbox_wrong(img, class_names[catID-1], res)
                            coco_obj = { "image_id"    :  step,
                                         "category_id" :  catID-1,
                                         "bbox"        :  [res[0], res[1], res[2]-res[0], res[3]-res[1]],
                                         "score"       :  res[4]}
                            coco_det.append(coco_obj)
                    else:
                        target_info=bbox_info[step][class_names[catID-1]]
                        for res in obj:
                            plot_bbox_right(img, class_names[catID-1], res, target_info)
                            coco_obj = { "image_id"    :  step,
                                         "category_id" :  catID-1,
                                         "bbox"        :  [res[0], res[1], res[2]-res[0], res[3]-res[1]],
                                         "score"       :  res[4]}
                            coco_det.append(coco_obj)
                

                img=img[:, :, [2, 1, 0]]
                cv2.imwrite(save_path+'/'+str(step)+'.png', np.clip(img*255, 0, 255).astype(np.uint8))

            create_video(save_path, save_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(save_path+'/Coco_det.txt', 'w') as f:
                json.dump(coco_det, f, indent=4, sort_keys=True)
            
            coco_target_file = save_path+'/Coco_target.txt'
            coco_det_file    = save_path+'/Coco_det.txt'
            cocoGt=COCO(coco_target_file)
            cocoDt=cocoGt.loadRes(coco_det_file)
            imgIds=sorted(cocoGt.getImgIds())
            imgIds=imgIds[0:100]
            imgId = imgIds[np.random.randint(100)]
            cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
            cocoEval.params.imgIds  = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()





if __name__=='__main__':

	parser = argparse.ArgumentParser()
    parser.add_argument("-Gaila_path",  dest="Gaila_path",  default=None, help="Path to Gaila Data File.")
    parser.add_argument("-MODEL_PATH",  dest="MODEL_PATH",  default=None, help="Path to Model.")
    parser.add_argument("-Task_ids",    dest="Task_ids",    default=None, help="Task_ids for Prediction.")
    parser.add_argument("-Scene_ids",   dest="Scene_ids",   default=None, help="Scene_ids for Prediction.")
    parser.add_argument("-SampleRate",  dest="SampleRate",  default=1,    help="SampleRate for Video.")
    parser.add_argument("-Vis_thresh",  dest="Vis_thresh",  default=0.3,  help="Vis_thresh for Filtering Results.")
    
    
    args = parser.parse_args()
    
	Gaila_Detection(Gaila_path   = args.Gaila_path, 
					Task_ids     = [args.Task_ids],
					Scene_ids    = [args.Scene_ids], 
					MODEL_PATH   = args.MODEL_PATH, 
					class_names  = args.class_names,  
					SampleRate   = args.SampleRate,
					Vis_thresh   = args.Vis_thresh)








