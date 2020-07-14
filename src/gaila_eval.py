"""
python3  ./gaila_eval.py gaila_ctdet --exp_id gaila_evalvis \
                                     --gpus 0 --vis_thresh 0.4 \
                                     --eval_vis_output ./TestEval/ \
                                     --load_annotations ~/scratch \
                                     --data_dir ~/scratch \
                                     --arch resdcn_18 --batch_size 32 --master_batch -1 \
                                     --load_model ../exp/gaila_ctdet/gaila_simplenet/model_last.pth

python gaila_eval.py gaila_ctdet --exp_id TEST_gaila_resdcn18_fpt400_ep10 --vis_thresh 0.4 --eval_vis_output ../exp/output_dump/ --load_annotations ../data  --arch resdcn_18 --load_model ../exp/gaila_ctdet/TRAIN_gaila_resdcn18_fpt400_ep10/model_last.pth
python gaila_eval.py gaila_ctdet --exp_id TEST_trainVS1_evalVS2_resdcn18 --vis_thresh 0.2 --eval_vis_output ../exp/TEST_trainVS1_evalVS2_resdcn18/output_dump/  --classnames_from ../reduced_classnames.txt --load_annotations ../data/eval_4_2b_task1 --load_predictions --arch resdcn_18 --load_model ../exp/gaila_ctdet/gaila_trainVS1_evalVS2_resdcn18/model_best.pth
python gaila_eval.py gaila_ctdet --exp_id TEST_trainVS1_evalVS2_resdcn18 --vis_thresh 0.4 --eval_vis_output ../exp/output_dump/ --frames_dir ~/data/GAILA/images_10hz/ --bounds_dir ~/data/GAILA/bounds/ --eval_pattern "4_2b_task1" --save_annotations ../data/eval_4_2b_task1  --arch resdcn_18 --load_model ../exp/gaila_ctdet/gaila_trainVS1_evalVS2_resdcn18/model_best.pth
python gaila_eval.py gaila_ctdet --exp_id TEST_trainVS1_evalVS1c_task123_resdcn18_fpt800 --vis_thresh 0.4   --classnames_from ../reduced_classnames.txt --load_annotations ~/scratch/TRAIN_trainVS1_evalVS1_resdcn18_fpt800/ --load_predictions --arch resdcn_18 --load_model ~/data/gzerveas/gaila_newer/exp/gaila_ctdet/TRAIN_trainVS1_evalVS1c_task123_resdcn18_fpt800/model_best.pth
python gaila_eval.py gaila_ctdet --exp_id TEST_trainVS1_evalVS2_resdcn18 --vis_thresh 0.2 --eval_vis_output ../exp/output_dump/ --frames_dir ~/data/GAILA/images_10hz/ --bounds_dir ~/data/GAILA/bounds/ --eval_pattern "4_2b_task1" --classnames_from ../reduced_classnames.txt --save_annotations ../data/eval_4_2b_task1 --arch resdcn_18 --load_model ../exp/gaila_ctdet/gaila_trainVS1_evalVS2_resdcn18/model_best.pth

python gaila_eval.py gaila_ctdet --exp_id FEAT_trainVS2_evalVS2_resdcn18_fpt100 --frames_dir ~/data/GAILA/images_10hz/ --bounds_dir ~/data/GAILA/bounds/ --eval_pattern "_2._" --classnames_from ../reduced_classnames.txt --save_annotations ~/scratch/FEAT_trainVS2_evalVS2_resdcn18_fpt100 --arch featresdcn_18 --batch_size 32 --master_batch -1 --lr 1.25e-4 --gpus 0 --num_workers 8  --load_model ~/data/gzerveas/gaila_newer/exp/gaila_ctdet/TRAIN_trainVS2_evalVS2_resdcn18_fpt800/model_last.pth --root_dir ~/data/gzerveas/gaila_newer/ --feat_extract ~/data/GAILA/obj_features/trainVS2_evalVS2_resdcn18_fpt100/ --frames_per_task 100 --no_visualization
python gaila_eval.py gaila_ctdet --exp_id TEST_trainMIX_evalVS2_resdcn18_fpt800 --frames_dir ~/data/GAILA/images_10hz/ --bounds_dir ~/data/GAILA/bounds/ --eval_pattern "_2._task[234]" --classnames_from ../reduced_classnames.txt --save_annotations ~/scratch/FEAT_task234_trainVS2_evalVS2_resdcn18 --arch featresdcn_18 --batch_size 32 --master_batch -1 --lr 1.25e-4 --gpus 0 --num_workers 8  --load_model ~/data/gzerveas/gaila_newer/exp/gaila_ctdet/TRAIN_trainVS2_evalVS2_resdcn18_fpt800/model_last.pth --root_dir ~/data/gzerveas/gaila_newer/ --feat_extract ~/data/GAILA/obj_features/task234_trainVS2_evalVS2_resdcn18/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import matplotlib.image as mpimg
import pickle

from opts import Opts
from logger import Logger
from utils.utils import AverageMeter, count_parameters
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory


def test(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    detector_task = 'ctdet'
    Dataset = dataset_factory['gaila']
    dataset = Dataset(opt, 'test')

    opt = Opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)
    Logger(opt)

    if opt.feat_extract:
        if not os.path.exists(opt.feat_extract):
            os.makedirs(opt.feat_extract)
        opt.load_predictions = False
        detector_task = 'ctdet_feat'
        feature_paths = []
        features = []
        heatmaps = []

    if opt.load_predictions:
        load_path = os.path.join(opt.save_dir, 'results.pickle')
        print("Loading detections from {} ... ".format(load_path), end="")
        with open(load_path, 'rb') as f:
            # The protocol version used is detected automatically
            results = pickle.load(f)
            print("Done")
    else:
        Detector = detector_factory[detector_task]
        detector = Detector(opt)
        results = {}

    video_path = {}
    metrics_pool = {}
    num_iters = len(dataset.all_frames)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for i in range(num_iters):
        img_id = int(dataset.all_frames[i][0].split('.')[0].split('/')[-1])
        img_path = dataset.all_frames[i][0]
        print(img_path)
        try:
            img = mpimg.imread(img_path)
        except:
            continue

        if not opt.load_predictions:
            ret = detector.run(img_path)
            if not opt.no_metrics:
                results[img_id] = ret['results']
        if opt.feat_extract:
            feature_path = img_to_feature_path(img_path)
            write_path = os.path.join(opt.feat_extract, feature_path)
            write_dir = os.path.split(write_path)[0]
            if not os.path.exists(write_dir):
                os.makedirs(write_dir)
            np.savez(write_path, features=ret['feature'], heatmaps=ret['heatmap'])
            if opt.feat_store_stacked:
                feature_paths.append(feature_path)
                features.append(ret['feature'])
                heatmaps.append(ret['heatmap'])

        if not (opt.no_visualization and opt.no_metrics):
            save_path, save_file, metrics = save_eval_vis(detRes=ret['results'], frameInfo=dataset.all_frames[i],
                                                          class_names=dataset.class_name, Vis_thresh=opt.vis_thresh,
                                                          save_path=opt.eval_vis_output)
            if save_path not in video_path:
                video_path[save_path] = [save_file]
            else:
                video_path[save_path].append(save_file)

            if save_path not in metrics_pool:
                metrics_pool[save_path] = [metrics]
            else:
                metrics_pool[save_path].append(metrics)

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(i, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        if not opt.load_predictions:
            for t in avg_time_stats:
                avg_time_stats[t].update(ret[t])
                Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        bar.next()
    bar.finish()

    # if not opt.load_predictions:
    #     write_path = os.path.join(opt.save_dir, 'detections.pickle')
    #     print("Serializing detections into {} ... ".format(write_path), end="")
    #     with open(write_path, 'wb') as f:
    #         pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    #     print("Done")

    if not opt.no_visualization:
        create_videos(video_path, opt.video_freq)
    if not opt.no_metrics:
        generate_basic_metrics(metrics_pool, opt.eval_vis_output)
        # This uses the COCO API to compute and present performance metrics, and dumps detections as json and pickle (if not loaded)
        dataset.run_eval(results, opt.save_dir, store_pickle=(not opt.load_predictions))

    if opt.feat_extract and opt.feat_store_stacked:
        feature_IDs = np.array(feature_paths)
        features = np.stack(features)
        heatmaps = np.stack(heatmaps)
        write_path = os.path.join(opt.feat_extract, 'all_features')
        print("Serializing features into {} ... ".format(write_path), end="")
        np.savez(write_path, feature_IDs=feature_IDs, features=features, heatmaps=heatmaps)
        print("Done")
    return


def img_to_feature_path(path):
    path = path.split('.')[0] + '.npz'  # swap image extension
    path = path.split('/')
    path = "/".join(path[-3:])  # paths look like this:  "path/to/frames_dir/9_1b/9_1b_task1/6142.png"
    return path


def save_eval_vis(detRes, frameInfo, class_names, Vis_thresh, save_path):
    Outpath = save_path + '/' + '/'.join(frameInfo[0].split('.')[0].split('/')[-3:-1]) + '/'
    imgId = frameInfo[0].split('.')[0].split('/')[-1]
    img = mpimg.imread(frameInfo[0])

    N = frameInfo[1].shape[0]
    tar_Info = {}
    tar_BbxAll = []
    for k in range(N):
        ctg_name = frameInfo[1]['name'].iloc[k]
        tar_bbx = [int(frameInfo[1]['topLeftX'].iloc[k]), int(frameInfo[1]['topLeftY'].iloc[k]),
                   int(frameInfo[1]['bottomRightX'].iloc[k]), int(frameInfo[1]['bottomRightY'].iloc[k])]
        tar_BbxAll.append(tar_bbx)
        if ctg_name not in tar_Info:
            tar_Info[ctg_name] = [tar_bbx]
        else:
            tar_Info[ctg_name].append(tar_bbx)

    IOU_sum_correct = 0
    IOU_sum_wrong = 0
    prob_sum_correct = 0
    prob_sum_wrong = 0
    cnt_correct = 0
    cnt_wrong = 0
    for catID, obj in detRes.items():
        if class_names[catID - 1] not in tar_Info:
            for res in obj:
                IOU_sum_wrong += get_IOU(res, tar_BbxAll)
                prob_sum_wrong += res[4]
                cnt_wrong += 1
                if res[4] > Vis_thresh:
                    plot_bbox_wrong(img, class_names[catID - 1], res)
        else:
            for res in obj:
                IOU_sum_correct += get_IOU(res, tar_Info[class_names[catID - 1]])
                prob_sum_correct += res[4]
                cnt_correct += 1
                if res[4] > Vis_thresh:
                    plot_bbox_right(img, class_names[catID - 1], res, tar_Info[class_names[catID - 1]])

    metrics = [IOU_sum_correct, IOU_sum_wrong,
               prob_sum_correct, prob_sum_wrong,
               cnt_correct, cnt_wrong]

    img = img[:, :, [2, 1, 0]]
    Outfile = Outpath + '/' + str(imgId) + '_post.png'
    if not opt.no_visualization:
        if not os.path.exists(Outpath):
            os.makedirs(Outpath)
        cv2.imwrite(Outfile, np.clip(img * 255, 0, 255).astype(np.uint8))
    return Outpath, Outfile, metrics


def create_videos(video_path, fr):
    for sp in video_path:
        if not os.path.exists(sp):
            os.makedirs(sp)
        imgs = []
        for file in sorted(video_path[sp], key=lambda x: int(x.split('/')[-1].split('_')[0])):
            imgs.append(cv2.imread(file))

        height, width, layers = imgs[1].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(sp + '/video.mp4', fourcc, fr, (width, height))

        for img in imgs:
            video.write(img)

        cv2.destroyAllWindows()
        video.release()


def generate_basic_metrics(metrics_pool, outpath):
    IOU_sum_correct = 0
    IOU_sum_wrong = 0
    prob_sum_correct = 0
    prob_sum_wrong = 0
    cnt_correct = 0
    cnt_wrong = 0

    for sp, metrics in metrics_pool.items():
        IOU_correct = sum([it[0] for it in metrics])
        IOU_wrong = sum([it[1] for it in metrics])
        prob_correct = sum([it[2] for it in metrics])
        prob_wrong = sum([it[3] for it in metrics])
        cnt_c = sum([it[4] for it in metrics])
        cnt_w = sum([it[5] for it in metrics])

        res = {'mean_IOU_correct': IOU_correct / cnt_c,
               'mean_IOU_wrong': IOU_wrong / cnt_w,
               'mean_IOU': (IOU_correct + IOU_wrong) / (cnt_c + cnt_w),
               'mean_prob_correct': prob_correct / cnt_c,
               'mean_prob_wrong': prob_wrong / cnt_w,
               'mean_prob': (prob_correct + prob_wrong) / (cnt_c + cnt_w)}
        if not os.path.exists(sp):
            os.makedirs(sp)
        with open(sp + '/basic_metrics.txt', 'w') as f:
            json.dump(res, f, indent=4, sort_keys=True)

        IOU_sum_correct += IOU_correct
        IOU_sum_wrong += IOU_wrong
        prob_sum_correct += prob_correct
        prob_sum_wrong += prob_wrong
        cnt_correct += cnt_c
        cnt_wrong += cnt_w

    res = {'mean_IOU_correct': IOU_sum_correct / cnt_correct,
           'mean_IOU_wrong': IOU_sum_wrong / cnt_wrong,
           'mean_IOU': (IOU_sum_correct + IOU_sum_wrong) / (cnt_correct + cnt_wrong),
           'mean_prob_correct': prob_sum_correct / cnt_correct,
           'mean_prob_wrong': prob_sum_wrong / cnt_wrong,
           'mean_prob': (prob_sum_correct + prob_sum_wrong) / (cnt_correct + cnt_wrong)}
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    with open(outpath + '/basic_metrics_all.txt', 'w') as f:
        json.dump(res, f, indent=4, sort_keys=True)


def plot_bbox_wrong(img, name, detector_bbox):
    txt = '{} p={:.3f}'.format(name, detector_bbox[4])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    c = (255, 0, 0)
    cv2.rectangle(img, (int(detector_bbox[0]), int(detector_bbox[1])), (int(detector_bbox[2]), int(detector_bbox[3])),
                  c, 2)
    cv2.rectangle(img,
                  (int(detector_bbox[0]), int(detector_bbox[1] - cat_size[1] - 2)),
                  (int(detector_bbox[0] + cat_size[0]), int(detector_bbox[1] - 2)), c, -1)
    cv2.putText(img, txt, (int(detector_bbox[0]), int(detector_bbox[1] - 2)),
                font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


def plot_bbox_right(img, name, detector_bbox, targetBoxes):
    IOU = get_IOU(detector_bbox, targetBoxes)

    txt = '{} p={:.3f} iou={:.2f}'.format(name, detector_bbox[4], IOU)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    c = (0, 255, 0)
    cv2.rectangle(img, (int(detector_bbox[0]), int(detector_bbox[1])), (int(detector_bbox[2]), int(detector_bbox[3])),
                  c, 2)
    cv2.rectangle(img,
                  (int(detector_bbox[0]), int(detector_bbox[1] - cat_size[1] - 2)),
                  (int(detector_bbox[0] + cat_size[0]), int(detector_bbox[1] - 2)), c, -1)
    cv2.putText(img, txt, (int(detector_bbox[0]), int(detector_bbox[1] - 2)),
                font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


def IOU(boxA, boxB):
    interAx = max(boxA[0], boxB[0])
    interAy = max(boxA[1], boxB[1])
    interBx = min(boxA[2], boxB[2])
    interBy = min(boxA[3], boxB[3])

    interArea = max(0, interBx - interAx+1) * max(0, interBy - interAy+1)
    A_Area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    B_Area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(A_Area + B_Area - interArea)
    return iou

def get_IOU(boxA, boxBs):
    if len(boxBs)==0:
        return 0
    iou = []
    for boxB in boxBs:
        iou.append(IOU(boxA, boxB))
    return max(iou)


if __name__ == '__main__':
    opt = Opts().parse()
    test(opt)
