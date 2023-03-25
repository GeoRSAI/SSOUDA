import random
import time
import warnings
import sys
import argparse
import shutil
import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('../..')
from common.utils.data import ForeverDataIterator
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger

sys.path.append('.')
from openset_class_learning import ImageClassifier, SupConLoss, Known_class_detection, Unknown_class_detection
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    train_transform = utils.TwoViewTransform()
    val_transform = utils.get_val_transform()
    print("train transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    len_source_loader = len(train_source_loader)
    len_target_loader = len(train_target_loader)
    min_n_batch = min(len_source_loader, len_target_loader)
    max_n_batch = max(len_source_loader, len_target_loader)
    print('min_n_batch: ', min_n_batch, ' max_n_batch：', max_n_batch)
    if min_n_batch != 0:
        args.iters_per_epoch = max_n_batch

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=True).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    tdl_loss_fn = SupConLoss(temperature=0.1).cuda()
    unknown_cst_loss_fn = Unknown_class_detection(num_classes, t1=args.t1, t2=args.t2).cuda()
    known_cst_loss_fn = Known_class_detection(threshold=args.tk).cuda()

    # resume from the best checkpoint
    if args.phase != 'train':
        print('load checkpoint: best model')
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # image classification test
    if args.phase == 'test':
        # acc1 = validate(test_loader, classifier, args)
        acc = utils.validate(val_loader, classifier, args, device)
        print("Classification Accuracy = {:0.4f}".format(acc))
        return

    # start training
    best_acc = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, tdl_loss_fn, unknown_cst_loss_fn, known_cst_loss_fn,
              optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        acc = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc > best_acc:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc = max(acc, best_acc)

    print("best_accu = {:.3f}".format(best_acc))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc = utils.validate(val_loader, classifier, args, device)
    print("test_accu = {:.3f}".format(acc))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          tdl_loss_fn: nn.Module, unknown_cst_loss_fn: nn.Module, known_cst_loss_fn: nn.Module, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Cst Loss', ':5.4f')
    target_tdl_losses = AverageMeter('Tdl Loss', ':5.4f')
    source_sdl_losses = AverageMeter('Sdl Loss', ':5.4f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [losses, source_sdl_losses, target_tdl_losses, trans_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i in range(args.iters_per_epoch):
        (x_s, _), labels_s = next(train_source_iter)
        (x_t_w, x_t_s), labels_t = next(train_target_iter)
        bsz = labels_s.shape[0]

        x_t = torch.cat([x_t_w, x_t_s], dim=0)
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)
        y_t_w, y_t_s = torch.split(y_t, [bsz, bsz], dim=0)

        if epoch > args.pretrain_epoch:
            unknown_cst_loss = unknown_cst_loss_fn(logits_s=y_t_s, logits_w=y_t_w).cuda()
            known_cst_loss = known_cst_loss_fn(logits_s=y_t_s, logits_w=y_t_w)
        else:
            unknown_cst_loss = torch.tensor([0]).cuda()
            known_cst_loss = torch.tensor([0]).cuda()

        # source domain discriminative loss
        sdl_loss = F.cross_entropy(y_s, labels_s)
        # target domain discriminative loss
        tdl_loss = tdl_loss_fn(f_t)
        # consistency self-training loss
        cst_loss = unknown_cst_loss + known_cst_loss

        loss = sdl_loss + args.w1 * tdl_loss + args.w2 * cst_loss

        losses.update(loss.item(), labels_s.size(0))
        trans_losses.update(cst_loss.item(), labels_s.size(0))

        target_tdl_losses.update(tdl_loss.item(), labels_s.size(0))
        source_sdl_losses.update(sdl_loss.item(), labels_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    '''
    python ssouda.py /your_path/SSOUDA_dataset/ -s UCMD -t NWPU -a resnet50 --epochs 60 --seed 1 --log logs/ucmd_nwpu
    '''
    parser = argparse.ArgumentParser(description='SSOUDA for Openset Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-s', '--source', help='source domain')
    parser.add_argument('-t', '--target', help='target domain')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')

    parser.add_argument('--pretrain-epoch', default=5, type=int, help='pretrain epoch for discriminative feature learning')
    parser.add_argument('--w1', default=0.3, type=float, help='parameter for tdl loss weight')
    parser.add_argument('--w2', default=0.5, type=float, help='parameter for cst loss weight')
    parser.add_argument('--t1', default=0.009, type=float, help='parameter for unknown class candidate')
    parser.add_argument('--t2', default=0.35, type=float, help='parameter for unknown class 2nd theshold')
    parser.add_argument('--tk', default=0.95, type=float, help='parameter for known class detection')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='dann',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)

