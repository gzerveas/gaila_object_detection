#!/bin/bash
#SBATCH --job-name="Isaac_BETTER"
#SBATCH --output=job-%j.txt
#SBATCH --error=job-%j.txt
#SBATCH --time=96:00:00
#SBATCH -n 1
#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=100G


source activate CenterNet

python3  ./gaila_eval.py gaila_ctdet --exp_id gaila_evalvis \
                                     --gpus 0 --vis_thresh 0.4 \
                                     --eval_vis_output ./TestEval/ \
                                     --load_annotations ~/scratch \
                                     --data_dir ~/scratch \
                                     --arch resdcn_18 --batch_size 32 --master_batch -1 \
                                     --load_model ../exp/gaila_ctdet/gaila_simplenet/model_last.pth