"""

As indicated in the model, the network used a pretrained intiailization
which is pretrained on ImageNet for 3 epochs only.

    We find that it is useful to warm-start any DML optimization with weights of a
    partly-trained a standard softmax classifier. It is important to not use weights
    of a net trained to completion, as this would result in information dissipation
    and as such defeat the purpose of pursuing DML in the first place. Hence, we
    initialize all models with the weights of a net trained on ImageNet
    (Russakovsky et al., 2015) for 3 epochs only. (page 8, section 4)

"""

import argparse
import os
import sys
import shutil
import time
import torch

import numpy                    as np
import torchvision.models       as models
import torchvision.transforms   as transforms
import torchvision.datasets     as datasets


import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from datasets                   import ImageNet, oxford_iiit_pet
from models                     import magnetInception
from tensorboardX               import SummaryWriter    as Logger
from util.torch_utils           import to_var, save_checkpoint, AverageMeter, accuracy
from util                       import magnet_loss
from torch.optim.lr_scheduler   import MultiStepLR
from IPython                    import embed

import torch.nn as nn
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main(args):
    curr_time = time.time()


    print("#############  Read in Database   ##############")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)


    print("Time taken:  {} seconds".format(time.time() - curr_time) )
    curr_time = time.time()

    print("######## Initiate Model and Optimizer   ##############")
    # Model - inception_v3 as specified in the paper
    # Note: This is slightly different to the model used by the paper,
    # however, the differences should be minor in terms of implementation and impact on results
    model   = Net()
    model   = torch.nn.DataParallel(model).cuda()


    # Criterion was not specified by the paper, it was assumed to be cross entropy (as commonly used)
    criterion = magnet_loss(D = 12, M = 4, alpha = 7.18).cuda()    # Loss function
    params  = list(model.parameters())                                      # Parameters to train


    # Optimizer -- the optimizer is not specified in the paper, and was ssumed to
    # be SGD. The parameters of the model were also not specified and were set
    # to commonly values used by pytorch (lr = 0.1, momentum = 0.3, decay = 1e-4)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    # The paper does not specify an annealing factor, we set it to 1.0 (no annealing)
    scheduler = MultiStepLR( optimizer,
                             milestones=list(range(0, args.num_epochs, 1)),
                             gamma=args.annealing_factor)


    print("Time taken:  {} seconds".format(time.time() - curr_time) )
    curr_time = time.time()

    print("#############  Start Training     ##############")
    total_step = len(train_loader)


    for epoch in range(0, args.num_epochs):


        if args.evaluate_only:         exit()
        if args.optimizer == 'sgd':    scheduler.step()

        logger.add_scalar("Misc/Epoch Number", epoch, epoch * total_step)
        train_step( model        = model,
                    train_loader = train_loader,
                    criterion    = criterion,
                    optimizer    = optimizer,
                    epoch        = epoch,
                    step         = epoch * total_step,
                    valid_loader = valid_loader)


        #
        # curr_loss, curr_wacc = eval_step(   model       = model,
        #                                     data_loader = valid_loader,
        #                                     criterion   = criterion,
        #                                     step        = epoch * total_step,
        #                                     datasplit   = "valid")

        # args = save_checkpoint(  model      = model,
        #                          optimizer  = optimizer,
        #                          curr_epoch = epoch,
        #                          curr_loss  = curr_loss,
        #                          curr_step  = (total_step * epoch),
        #                          args       = args,
        #                          curr_acc   = curr_wacc,
        #                          filename   = ('model@epoch%d.pkl' %(epoch)))

    # Final save of the model
    args = save_checkpoint(  model      = model,
                             optimizer  = optimizer,
                             curr_epoch = epoch,
                             curr_loss  = curr_loss,
                             curr_step  = (total_step * epoch),
                             args       = args,
                             curr_acc   = curr_wacc,
                             filename   = ('model@epoch%d.pkl' %(epoch)))

def train_step(model, train_loader, criterion, optimizer, epoch, step, valid_loader = None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        # input = input
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)


        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        # loss = F.nll_loss(output, target_var)


        # measure accuracy and record loss

        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        model.zero_grad()

        loss.backward()
        optimizer.step()

        # measure elapsed time
        del loss, output
        batch_time.update(time.time() - end)

        if i % args.log_rate == 0 and i > 0:
            # print('Epoch: [{0}][{1}/{2}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
            #        epoch, i, len(train_loader), batch_time=batch_time,
            #        data_time=data_time, loss=losses))


            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  # 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


            # Log items
            logger.add_scalar("Misc/batch time (s)",    batch_time.avg,                                   step + i)
            logger.add_scalar("Misc/Train_%",           1.0 - (data_time.avg/batch_time.avg),             step + i)
            logger.add_scalar("Misc/epoch time (min)",  batch_time.sum / 60.,                             step + i)
            logger.add_scalar("Misc/time left (min)",   ((len(train_loader) - i) * batch_time.avg) / 60., step + i)
            logger.add_scalar(args.dataset + "/Loss train  (avg)",          losses.avg,                      step + i)
            # logger.add_scalar(args.dataset + "/Perc5 train (avg)",          top5.avg,                      step + i)
            # logger.add_scalar(args.dataset + "/Perc1 train (avg)",          top1.avg,                      step + i)

        end = time.time()

        # if valid_loader != None and i % args.eval_step == 0 and i > 0:
        #     _, _ = eval_step(   model       = model,
        #                         data_loader = valid_loader,
        #                         criterion   = criterion,
        #                         step        = step + i,
        #                         datasplit   = "valid")
        #     model.train()


def eval_step( model, data_loader,  criterion, step, datasplit):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var, -1)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_rate == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   i, len(data_loader), batch_time=batch_time, loss=losses)
            )


    logger.add_scalar(args.dataset + "/Loss valid  (avg)",   losses.avg, step)
    # logger.add_scalar(args.dataset + "/Perc5 valid (avg)",   top5.avg,   step)
    # logger.add_scalar(args.dataset + "/Perc1 valid (avg)",   top1.avg,   step)

    return losses.avg, 0.0


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # logging parameters
    parser.add_argument('--save_epoch',         type=int , default=10)
    parser.add_argument('--eval_epoch',         type=int , default=1)
    parser.add_argument('--eval_step',          type=int , default=1000)
    parser.add_argument('--log_rate',           type=int, default=10)
    parser.add_argument('--workers',            type=int, default=7)
    parser.add_argument('--world_size',         type=int, default=1)

    # training parameters
    parser.add_argument('--num_epochs',         type=int,   default=3)
    parser.add_argument('--batch_size',         type=int,   default=256)
    parser.add_argument('--lr',                 type=float, default=0.1)
    parser.add_argument('--momentum',           type=float, default=0.9)
    parser.add_argument('--weight_decay',       type=float, default=1e-4)
    parser.add_argument('--optimizer',          type=str,   default='sgd')
    parser.add_argument('--annealing_factor',   type=float, default=1.0)

    # experiment details
    parser.add_argument('--dataset',            type=str, default='oxford')
    parser.add_argument('--model',              type=str, default='inception')
    parser.add_argument('--experiment_name',    type=str, default= 'Test')
    parser.add_argument('--evaluate_only',      action="store_true",default=False)
    parser.add_argument('--evaluate_train',     action="store_true",default=False)
    parser.add_argument('--resume',             type=str, default=None)

    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

    # Magnet Loss Parameters
    # parser.add_argument('--M',                  type=int, default=12)       # Number of nearest clusters per mini-batch
    # parser.add_argument('--K',                  type=int, default=12)       # Number of clusters per class
    # parser.add_argument('--D',                  type=int, default=12)       # Number of examples per cluster


    args = parser.parse_args()
    print(args)
    print("")

    if args.dataset == "oxford":
        args.data_path = "/z/home/mbanani/datasets/Oxford-IIIT_Pet/"

    root_dir                    = os.path.dirname(os.path.abspath(__file__))
    experiment_result_dir       = os.path.join(root_dir, os.path.join('experiments',args.dataset))
    args.full_experiment_name   = ("exp_%s_%s_%s" % ( time.strftime("%m_%d_%H_%M_%S"), args.dataset, args.experiment_name) )
    args.experiment_path        = os.path.join(experiment_result_dir, args.full_experiment_name)
    args.best_loss              = sys.float_info.max
    args.best_acc               = 0.

    # Create model directory
    if not os.path.exists(experiment_result_dir):
        os.makedirs(experiment_result_dir)
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)

    print("Experiment path is : ", args.experiment_path)


    # Define Logger
    tensorboard_logdir = '/z/home/mbanani/tensorboard_logs'
    log_name    = args.full_experiment_name
    logger      = Logger(log_dir = os.path.join(tensorboard_logdir, log_name))

    main(args)
