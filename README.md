# Object Localization for Virtual Reality Environments
This repo contains the code for object detection in VR environments. The method is based on CenterNet and is applied on GAILA dataset.
![](readme/fig2.png)
> [**Objects as Points**](http://arxiv.org/abs/1904.07850),            
> Xingyi Zhou, Dequan Wang, Philipp Kr&auml;henb&uuml;hl,              


George Zerveas, Reza Esfandiarpoor, Zhizhong Chen

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions.

## Use

To build the dataset and convert the raw data to proper format run the following:

~~~
python main.py gaila_ctdet --exp_id TEST_gaila_simplenet  --frames_dir ~/data/GAILA/images_10hz/ --bounds_dir ~/data/GAILA/bounds/ --save_annotations ~/scratch --arch simple --batch_size 32 --master_batch -1 --lr 1.25e-4 --gpus 0 --num_workers 4  --frames_per_task 20 --num_epochs 2 --debug 4
~~~

For training the model run:   

~~~
python  main.py gaila_ctdet --exp_id TRAIN_gaila_resdcn18_fpt400_ep10  --load_annotations ../data/ --arch resdcn_18 --batch_size 128 --master_batch -1 --lr 1.25e-4 --gpus 0 --num_workers 8 --frames_per_task 400 --num_epochs 100 --resume
~~~

For testing the model run:

~~~
python gaila_eval.py gaila_ctdet --exp_id TEST_gaila_resdcn18_fpt400_ep10 --vis_thresh 0.4 --eval_vis_output ../exp/output_dump/ --load_annotations ../data --batch_size 32 --num_workers 8 --master_batch -1 --arch resdcn_18 --load_model ../exp/gaila_ctdet/TRAIN_gaila_resdcn18_fpt400_ep10/model_last.pth
~~~
