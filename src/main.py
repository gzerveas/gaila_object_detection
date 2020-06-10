"""
python main.py gaila_ctdet --exp_id TRAIN_trainVS2_evalVS2_resdcn18_fpt800 --frames_dir ~/data/GAILA/images_10hz/ --bounds_dir ~/data/GAILA/bounds/ --train_pattern "_2._task[^1]" --eval_pattern "_2b_task1" --classnames_from ../reduced_classnames.txt --save_annotations ~/scratch/TRAIN_trainVS2_evalVS1_resdcn18_fpt800 --arch resdcn_18 --batch_size 32 --master_batch -1 --lr 1.25e-4 --gpus 0 --num_workers 8  --frames_per_task 800 --resume --num_epochs 30
python main.py gaila_ctdet --exp_id TRAIN_trainVS2_evalVS1_resdcn18_fpt800 --frames_dir ~/data/GAILA/images_10hz/ --bounds_dir ~/data/GAILA/bounds/ --train_pattern "_2._" --eval_pattern "_1b_task1" --classnames_from ../reduced_classnames.txt --save_annotations ~/scratch/TRAIN_trainVS2_evalVS1_resdcn18_fpt800 --arch resdcn_18 --batch_size 32 --master_batch -1 --lr 1.25e-4 --gpus 0 --num_workers 8  --frames_per_task 800 --resume --num_epochs 30
python main.py gaila_ctdet --exp_id TRAIN_trainVS2_evalVS1_resdcn18_fpt800  --classnames_from ../reduced_classnames.txt --load_annotations ~/scratch/trainVS2_evalVS1/ --arch resdcn_18 --batch_size 32 --master_batch -1 --lr 1.25e-4 --gpus 0 --num_workers 8  --frames_per_task 800 --resume
python main.py gaila_ctdet --exp_id TRAIN_trainMIX_evalVS2_resdcn18_fpt800 --frames_dir ~/data/GAILA/images_10hz/ --bounds_dir ~/data/GAILA/bounds/ --train_pattern "(_1._)|(_task[^1]_)|(_2[ac]_task1)" --eval_pattern "_2b_task1" --classnames_from ../reduced_classnames.txt --save_annotations ~/scratch/TRAIN_trainMIX_evalVS2_resdcn18_fpt800 --arch resdcn_18 --batch_size 32 --master_batch -1 --lr 1.25e-4 --gpus 0 --num_workers 8  --frames_per_task 800 --resume --num_epochs 30
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import json

import torch
import torch.utils.data
from opts import Opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from utils.utils import count_parameters


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    val_dataset = Dataset(opt, 'val')
    opt = Opts().update_dataset_info_and_set_heads(opt, val_dataset)
    print(opt)

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    print("Model:")
    print(model)
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))


    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    if opt.test:
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        print("{} images failed!".format(len(val_loader.dataset.failed_images)))
        print(val_loader.dataset.failed_images)
        return

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Report failed images
    failed_images = list(train_loader.dataset.failed_images | val_loader.dataset.failed_images)
    if len(failed_images):
        print("{} images failed!".format(len(failed_images)))
        dump_path = os.path.join(opt.save_dir, 'failed_images.json')
        with open(dump_path, 'w') as f:
            json.dump(failed_images, f, sort_keys=True, indent=4, separators=(',', ': '))
        print("Failed image paths saved in: {}".format(dump_path))


    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    opt = Opts().parse()
    main(opt)
