from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import random
import glob
from tqdm import tqdm
import re
import matplotlib.image as mpimg
import pandas as pd
import pickle

import torch.utils.data as data


class GAILA(data.Dataset):

    # Canonical setting
    num_classes = 36  # class variable is used if no instance member variable is set
    class_exceptions = ['Wall', 'Ceiling', 'Floor']
    ########## NEED TO COMPUTE ##############
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split, shuffle=True):

        super(GAILA, self).__init__()
        self.failed_images = set([])
        self.max_objs = 64  # 128
        # threshold of size proportion in x, y that the target bounding boxes must have within the image frame in order to be kept
        self.threshold = 0.3
        ########## KEPT TEMPORARILY ##############
        self._data_rng = np.random.RandomState(123) # GEO: needed with no_color_aug
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], # GEO: needed with no_color_aug
                                 dtype=np.float32)
        self._eig_vec = np.array([ # GEO: needed with no_color_aug
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.img_shape = [960, 540]  # size of each frame
        ########## KEPT TEMPORARILY ##############

        self.split = split  # which split of the dataset we are looking at
        self.opt = opt

        if opt.classnames_from:
            self.class_name = sorted([line.rstrip() for line in open(opt.classnames_from).readlines()])
            print("{} classes in file '{}': {}".format(len(self.class_name), opt.classnames_from, self.class_name))
            class_names_set = set(self.class_name)
        else:
            self.class_name = None

        if opt.load_annotations is not None:
            load_path = os.path.join(opt.load_annotations, 'frame_annotations_{}.pickle'.format(self.split))
            print("Loading annotations from {} ... ".format(load_path), end="")
            with open(load_path, 'rb') as f:
                # The protocol version used is detected automatically
                self.all_frames = pickle.load(f)
                print("Done")

        else:
            print('Building GAILA {} dataset ...'.format(self.split))
            # Select directories for training and evaluation
            task_dirs = glob.glob(os.path.join(opt.frames_dir, '*/*'))  # list of all task directories
            if len(task_dirs) == 0:
                raise Exception('No task directories found using: {}'.format(os.path.join(opt.frames_dir, '*/*')))

            if opt.eval_pattern is None:
                # by default evaluate on half the tasks of an given scene (configuration) for both visual styles (blocky, photorealistic)
                eval_pattern = r'_1c_task[123]|_2c_task[456]'
            else:
                eval_pattern = opt.eval_pattern

            if self.split != 'train':
                selected_dirs = list(filter(lambda x: re.search(eval_pattern, x), task_dirs))
            else:  # for training set
                if opt.train_pattern is None:  # use the complement of eval set
                    selected_dirs = list(filter(lambda x: not re.search(eval_pattern, x), task_dirs))
                else:
                    selected_dirs = list(filter(lambda x: re.search(opt.train_pattern, x), task_dirs))

            self.all_frames = []  # list of tuples: (frame path, frame dataframe containing annotations)
            for _dir in tqdm(selected_dirs, desc=f'Loading {split} annotation files'):
                image_paths = os.listdir(_dir)  # base file names of all frame files corresponding to task directory
                image_ids = []  # list of image IDs, by cutting extension from path
                for img in image_paths:
                    full_path = os.path.join(_dir, img)
                    if os.stat(full_path).st_size > 10:  # check that the frame file is not corrupt (size > 10 bytes)
                        image_ids.append(int(img.split('.')[0]))
                    else:
                        self.failed_images.add(full_path)

                bbox_path = os.path.join(opt.bounds_dir, os.path.basename(_dir) + '_bounds.txt')
                raw_path = _dir.replace('images_10hz', 'raw') + '.txt'
                # Check that data file exists
                if not os.path.exists(bbox_path):
                    print("ERROR: {} not found!".format(bbox_path))
                    continue
                if not os.path.exists(raw_path):
                    print("ERROR: {} not found!".format(raw_path))
                    continue

                with open(bbox_path, 'r') as f:
                    lines = f.readlines()
                    bbox_frame = pd.DataFrame([json.loads(line.rstrip()) for line in lines[:-1]])  # this excludes last (malformed) line (missing 1 object)
                with open(raw_path, 'r') as f:
                    lines = f.readlines()
                    raw_frame = pd.DataFrame([json.loads(line.rstrip()) for line in lines])
                # Exclude the last faulty frame
                bbox_frame = bbox_frame[bbox_frame['step'] != bbox_frame.iloc[-1]['step']]
                # Drop annotations for objects that are not visible
                raw_frame = raw_frame[raw_frame['visible'] == True]
                bbox_index = bbox_frame.set_index('step').index
                raw_index = raw_frame.set_index('step').index
                bbox_frame = bbox_frame[bbox_index.isin(raw_index)]
                # Drop annotations for frames not existing on disk
                bbox_frame = bbox_frame[bbox_frame['step'].isin(image_ids)]
                # Drop objects in exceptions list or not in pre-specified class name list
                if self.class_name is None:
                    bbox_frame = bbox_frame[~bbox_frame.name.isin(GAILA.class_exceptions)]
                else:
                    bbox_frame = bbox_frame[bbox_frame.name.isin(class_names_set)]
                bbox_frame = list(bbox_frame.groupby('step'))  # list of step/frame groups
                random.shuffle(bbox_frame)
                selected_frames = bbox_frame[:opt.frames_per_task]
                selected_frames = [(os.path.join(_dir, str(i) + '.png'), self._filter_bboxes(j)) for i, j in selected_frames]
                self.all_frames.extend(selected_frames)

            if opt.save_annotations is not None:
                if not os.path.exists(opt.save_annotations):
                    os.makedirs(opt.save_annotations)
                write_path = os.path.join(opt.save_annotations, 'frame_annotations_{}.pickle'.format(self.split))
                print("Serializing annotations into {} ... ".format(write_path), end="")
                with open(write_path, 'wb') as f:
                    pickle.dump(self.all_frames, f, pickle.HIGHEST_PROTOCOL)
                print("Done")

        if shuffle:
            random.shuffle(self.all_frames)

        if opt.classnames_from is None:  # Infer classes from loaded annotations
            dataframes = pd.concat(annotations for path, annotations in self.all_frames)
            self.class_name = sorted(list(dataframes.name.unique()))
            print("{} classes extracted: {}".format(len(self.class_name), self.class_name))
            if opt.save_classnames_to:
                with open(opt.save_classnames_to, 'w') as f:
                    for name in self.class_name:
                        f.write(name + '\n')

        self.cat_ids = {name: ind for ind, name in enumerate(self.class_name)}
        self.num_classes = len(self.class_name)
        self.num_samples = len(self.all_frames)

        # Converts dataset (image name, bounding boxes and class names of objects) to COCO format and dumps to JSON
        # This is used eventually for computing performance metrics
        coco_path = COCO_anno(frameInfoBox=self.all_frames,
                              class_names=self.class_name,
                              save_path=opt.eval_vis_output)
        self.coco = coco.COCO(coco_path)

        print('Loaded {} frames/samples for {}'.format(self.num_samples, split))


    def _coco_box_to_bbox(self, bbox):
        """
        logic: if (x < -0.8*box_width) or (x > W - 0.2*box_width) or (y < -0.8*box_height) or (y > H - 0.2*box_height):
                    ignore object
                else:
                    redefine (crop) box to fit inside the image
        :param bbox:  [x, y, bbox_width, bbox_height], where (x,y) the coordinates of the top left corner of the box
        :param img_shape: [image_width, image_height]
        :param threshold:
        :return: None if object is filtered out,
                otherwise same or cropped [x, y, bbox_width, bbox_height]
        """

        top_left = (bbox[0], bbox[1])
        bottom_right = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        _bbox = (top_left, bottom_right)
        if top_left[0] <= 0 or top_left[1] <= 0:
            visible_width = bbox[2]
            visible_height = bbox[3]

            if top_left[0] < 0:
                visible_width = visible_width + top_left[0]
            if top_left[1] < 0:
                visible_height = visible_height + top_left[1]

            if visible_height < self.threshold * bbox[3] or visible_width < self.threshold * bbox[2]:
                return None
            else:
                top_left = (bottom_right[0] - visible_width + 1, bottom_right[1] - visible_height + 1)
                _bbox = (top_left, bottom_right)
        if bottom_right[0] >= self.img_shape[1] or bottom_right[1] >= self.img_shape[0]:
            visible_width = bbox[2]
            visible_height = bbox[3]

            if bottom_right[0] >= self.img_shape[1]:
                visible_width = self.img_shape[1] - top_left[0]
            if bottom_right[1] >= self.img_shape[0]:
                visible_height = self.img_shape[0] - top_left[1]

            if visible_height < self.threshold * bbox[3] or visible_width < self.threshold * bbox[2]:
                return None
            else:
                bottom_right = (top_left[0] + visible_width - 1, top_left[1] + visible_height - 1)
                _bbox = (top_left, bottom_right)
        if _bbox is not None:
            _bbox = np.array([_bbox[0][0], _bbox[0][1], _bbox[1][0], _bbox[1][1]],
                            dtype=np.int32).astype(np.float32)
        return _bbox

    # GEO: use dataframe.apply for this
    def _filter_bboxes(self, dataframe):
        object_list = list()
        num_objects = len(dataframe)
        for i in range(num_objects):
            input_bbox = dataframe.iloc[i]
            bbox = [input_bbox['pixelPosX'], input_bbox['pixelPosY'], input_bbox['pixelWidth'], input_bbox['pixelHeight']]
            new_bbox = self._coco_box_to_bbox(bbox)
            if new_bbox is not None:
                input_bbox.at["pixelPosX"] = new_bbox[0]
                input_bbox.at["pixelPosY"] = new_bbox[1]
                input_bbox.at["pixelWidth"] = new_bbox[2]
                input_bbox.at["pixelHeight"] = new_bbox[3]
                object_list.append(input_bbox)

        df = pd.DataFrame(object_list)
        df = df.rename(columns={"pixelPosX": "topLeftX", "pixelPosY": "topLeftY", "pixelWidth": "bottomRightX", "pixelHeight": "bottomRightY"})
        return df


    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = cls_ind #self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        """Converts model detections to custom format and dumps to JSON file"""
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    @staticmethod
    def pickle_results(results, save_dir):
        """This is used to enable drawing bounding boxes and computing other metrics in gaila_eval.py"""
        write_path = os.path.join(save_dir, 'results.pickle')
        print("Serializing detections into {} ... ".format(write_path), end="")
        with open(write_path, 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        print("Done")

    def run_eval(self, results, save_dir, store_pickle=True):
        """Converts detections to COCO format and runs COCO API evaluation"""

        if store_pickle:
            self.pickle_results(results, save_dir)

        # this is done only because there seems to be no other way to initialize COCOeval than load from file
        self.save_results(results, save_dir)
        coco_detections = self.coco.loadRes('{}/results.json'.format(save_dir))
        # Uses COCO API to calculate and present performance metrics
        coco_eval = COCOeval(self.coco, coco_detections, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


def COCO_anno(frameInfoBox, class_names, save_path):
    ctg_dict = {}
    for k in range(len(class_names)):
        ctg_dict[class_names[k]] = k + 1

    coco_file = {}
    coco_file["info"] = {"description": "GAILA",
                         "url": "dummy",
                         "version": "1.0",
                         "year": 2020,
                         "contributor": "dummy",
                         "date_created": "2020/05/05"}

    coco_file['licenses'] = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                              "id": 1,
                              "name": "Attribution-NonCommercial-ShareAlike License"}]
    coco_file["images"] = []
    coco_file["annotations"] = []
    coco_file["categories"] = []

    for k in range(len(class_names)):
        ctg_info = {"supercategory": class_names[k],
                    "id": 1 + k,
                    "name": class_names[k]}
        coco_file["categories"].append(ctg_info)

    H = 0
    W = 0
    for Info in frameInfoBox:
        try:
            img = mpimg.imread(Info[0])
            H = img.shape[0]
            W = img.shape[1]
            break
        except:
            continue

    cnt = 1
    for frameInfo in frameInfoBox:
        imgId = frameInfo[0].split('.')[0].split('/')[-1]
        img_info = {"license": 1,
                    "file_name": imgId + '.png',
                    "coco_url": "dummy_url",
                    "height": H,
                    "width": W,
                    "date_captured": "2020-02-02 02:02:02",
                    "flickr_url": "dummy_url",
                    "id": int(imgId)}
        coco_file["images"].append(img_info)

        anno_info = []
        N = frameInfo[1].shape[0]
        for k in range(N):
            ctg_name = frameInfo[1]['name'].iloc[k]
            tar_bbx = [int(frameInfo[1]['topLeftX'].iloc[k]),
                       int(frameInfo[1]['topLeftY'].iloc[k]),
                       int(frameInfo[1]['bottomRightX'].iloc[k] - frameInfo[1]['topLeftX'].iloc[k]),
                       int(frameInfo[1]['bottomRightY'].iloc[k] - frameInfo[1]['topLeftY'].iloc[k])]

            anno = {"segmentation": [],
                    "area": tar_bbx[2] * tar_bbx[3],
                    "iscrowd": 0,
                    "image_id": int(imgId),
                    "bbox": tar_bbx,
                    "category_id": ctg_dict[ctg_name],
                    "id": cnt}
            cnt += 1
            anno_info.append(anno)
        coco_file["annotations"].extend(anno_info)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + '/Gaila_Annotation.json', 'w') as f:
        json.dump(coco_file, f)
    return save_path + '/Gaila_Annotation.json'