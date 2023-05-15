
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import ast
import torch
import random
import argparse
import numpy as np

from data_loader.loader import Loader
from core import Base, train, test
from tools import make_dirs, Logger, os_walk, time_now
import warnings
warnings.filterwarnings("ignore")

best_mAP = 0
best_rank1 = 0
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    global best_mAP
    global best_rank1

    loaders = Loader(config)
    model = Base(config)

    make_dirs(model.output_path)
    make_dirs(model.save_model_path)
    make_dirs(model.save_logs_path)

    logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'log.txt'))
    logger('\n' * 3)
    logger(config)

    if config.mode == 'train':
        if config.resume_train_epoch >= 0:
            model.resume_model(config.resume_train_epoch)
            start_train_epoch = config.resume_train_epoch
        else:
            start_train_epoch = 0

        if config.auto_resume_training_from_lastest_step:
            root, _, files = os_walk(model.save_model_path)
            if len(files) > 0:
                indexes = []
                for file in files:
                    indexes.append(int(file.replace('.pth', '').split('_')[-1]))
                indexes = sorted(list(set(indexes)), reverse=False)
                model.resume_model(indexes[-1])
                start_train_epoch = indexes[-1]
                logger('Time: {}, automatically resume training from the latest step (model {})'.format(time_now(),
                                    indexes[-1]))

        for current_epoch in range(start_train_epoch, config.total_train_epoch):
            model.rgb_model_lr_scheduler.step(current_epoch)
            model.ir_model_lr_scheduler.step(current_epoch)
            model.shared_model_lr_scheduler.step(current_epoch)
            model.classifier_lr_scheduler.step(current_epoch)

            if current_epoch < config.total_train_epoch:
                _, result = train(model, loaders, config)
                logger('Time: {}; Epoch: {}; {}'.format(time_now(), current_epoch, result))

            if current_epoch + 1 >= 70 and (current_epoch + 1) % config.eval_epoch == 0:
                cmc, mAP, mINP = test(model, loaders, config)
                is_best_rank = (cmc[0] >= best_rank1)
                best_rank1 = max(cmc[0], best_rank1)
                model.save_model(current_epoch, is_best_rank)
                logger('Time: {}; Test on Dataset: {}, \nmINP: {} \nmAP: {} \n Rank: {}'.format(time_now(),
                                                                                            config.dataset,
                                                                                            mINP, mAP, cmc))

    elif config.mode == 'test':
        model.resume_model(config.resume_test_model)
        cmc, mAP, mINP = test(model, loaders, config)
        logger('Time: {}; Test on Dataset: {}, \nmINP: {} \nmAP: {} \n Rank: {}'.format(time_now(),
                                                                                       config.dataset,
                                                                                       mINP, mAP, cmc))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--test_mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--gall_mode', default='single', type=str, help='single or multi')
    parser.add_argument('--regdb_test_mode', default='v-t', type=str, help='')
    parser.add_argument('--module', type=str, default='B_tri', help='B')
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
    parser.add_argument('--sysu_data_path', type=str, default='/data/ls/reid_data/SYSU-MM01/')
    parser.add_argument('--regdb_data_path', type=str, default='/opt/data/private/data/RegDB/')
    parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')
    parser.add_argument('--batch-size', default=32, type=int, metavar='B', help='training batch size')
    parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pid_num', type=int, default=395)
    parser.add_argument('--in_dim', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=0.00035)
    parser.add_argument('--c_learning_rate', type=float, default=0.0007)
    parser.add_argument('--num_pos', default=4, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--lower', type=float, default=0.02)
    parser.add_argument('--upper', type=float, default=0.4)
    parser.add_argument('--ratio', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    parser.add_argument('--output_path', type=str, default='sysu/base/',
                        help='path to save related informations')
    parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')
    parser.add_argument('--resume_train_epoch', type=int, default=-1, help='-1 for no resuming')
    parser.add_argument('--auto_resume_training_from_lastest_step', type=ast.literal_eval, default=False)
    parser.add_argument('--total_train_epoch', type=int, default=120)
    parser.add_argument('--eval_epoch', type=int, default=2)
    parser.add_argument('--resume_test_model', type=int, default=119, help='-1 for no resuming')

    config = parser.parse_args()
    seed_torch(config.seed)
    main(config)
