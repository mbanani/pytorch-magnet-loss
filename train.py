import argparse
import os
import sys
import shutil
import time

import numpy as np

import torch

from util                       import ViewpointLoss, Logger, Paths
from util                       import get_data_loaders
from util.torch_utils           import to_var, save_checkpoint
from torch.optim.lr_scheduler   import MultiStepLR

def main(args):
    curr_time = time.time()

    print("#############  Read in Database   ##############")
    # Data loading code (From PyTorch example https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    data_path = '/z/dat/ImageNet_2012'
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'validation')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print("Time taken:  {} seconds".format(time.time() - curr_time) )
    curr_time = time.time()

    print("######## Initiate Model and Optimizer   ##############")

    print("Time taken:  {} seconds".format(time.time() - curr_time) )
    curr_time = time.time()

    criterion = 0                               # Loss Function
    params  = list(model.parameters())          # Parameters to train


    # Optimizer
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum = args.momentum, weight_decay = 0.0)
    scheduler = MultiStepLR( optimizer,
                             milestones=list(range(0, args.num_epochs, 1)),
                             gamma=args.annealing_factor)


    # Train on GPU if available
    if torch.cuda.is_available():
        model.cuda()


    print("Time to initialize take: ", time.time() - initialization_time)
    print("#############  Start Training     ##############")
    total_step = len(train_loader)

    for epoch in range(0, args.num_epochs):

        if epoch % args.eval_epoch == 0:
            if args.evaluate_train:
                _, _ = eval_step(   model       = model,
                                    data_loader = train_loader,
                                    criterion   = criterion,
                                    step        = epoch * total_step,
                                    datasplit   = "train")


            curr_loss, curr_wacc = eval_step(   model       = model,
                                                data_loader = test_loader,
                                                criterion   = criterion,
                                                step        = epoch * total_step,
                                                datasplit   = "test")

            if valid_loader != None:
                curr_loss, curr_wacc = eval_step(   model       = model,
                                                    data_loader = valid_loader,
                                                    criterion   = criterion,
                                                    step        = epoch * total_step,
                                                    datasplit   = "valid")



        if args.evaluate_only:
            exit()

        if epoch % args.save_epoch == 0 and epoch > 0:

            args = save_checkpoint(  model      = model,
                                     optimizer  = optimizer,
                                     curr_epoch = epoch,
                                     curr_step  = (total_step * epoch),
                                     args       = args,
                                     curr_loss  = curr_loss,
                                     curr_acc   = curr_wacc,
                                     filename   = ('model@epoch%d.pkl' %(epoch)))

        if args.optimizer == 'sgd':
            scheduler.step()

        logger.add_scalar_value("Misc/Epoch Number", epoch, step=epoch * total_step)
        train_step( model        = model,
                    train_loader = train_loader,
                    criterion    = criterion,
                    optimizer    = optimizer,
                    epoch        = epoch,
                    step         = epoch * total_step,
                    valid_loader = valid_loader,
                    valid_type   = "valid")

    # Final save of the model
    args = save_checkpoint(  model      = model,
                             optimizer  = optimizer,
                             curr_epoch = epoch,
                             curr_step  = (total_step * epoch),
                             args       = args,
                             curr_loss  = curr_loss,
                             curr_acc   = curr_wacc,
                             filename   = ('model@epoch%d.pkl' %(epoch)))

def train_step(model, train_loader, criterion, optimizer, epoch, step, valid_loader = None, valid_type = "valid"):
    model.train()
    total_step      = len(train_loader)
    epoch_time      = time.time()
    batch_time      = time.time()
    processing_time = 0
    loss_sum        = 0.
    counter         = 0

    for i, (images, label) in enumerate(train_loader):
        counter = counter + 1
        training_time = time.time()

        # Set mini-batch dataset
        images  = to_var(images, volatile=False)
        label   = to_var(label)

        # Forward, Backward and Optimize
        model.zero_grad()

        pred = model(images)

        loss = criterion(pred, label)

        loss_sum += loss.data[0]

        loss.backward()
        optimizer.step()

        # Log losses
        logger.add_scalar_value("(" + args.dataset + ") Loss", loss.data[0] , step=step + i)

        processing_time += time.time() - training_time

        # Print log info
        if i % args.log_rate == 0 and i > 0:
            time_diff = time.time() - batch_time

            curr_batch_time = time_diff / (1.*args.log_rate)
            curr_train_per  = processing_time/time_diff
            curr_epoch_time = (time.time() - epoch_time) * (total_step / (i+1.))
            curr_time_left  = (time.time() - epoch_time) * ((total_step - i) / (i+1.))

            print("Epoch [%d/%d] Step [%d/%d]: Training Loss = %2.5f, Batch Time = %.2f sec, Time Left = %.1f mins." %( epoch, args.num_epochs,
                                                                                                                        i, total_step,
                                                                                                                        loss_sum / float(counter),
                                                                                                                        curr_batch_time,
                                                                                                                        curr_time_left / 60.))

            logger.add_scalar_value("Misc/batch time (s)",    curr_batch_time,        step=step + i)
            logger.add_scalar_value("Misc/Train_%",           curr_train_per,         step=step + i)
            logger.add_scalar_value("Misc/epoch time (min)",  curr_epoch_time / 60.,  step=step + i)
            logger.add_scalar_value("Misc/time left (min)",   curr_time_left / 60.,   step=step + i)

            # Reset counters
            counter = 0
            loss_sum = 0.
            processing_time = 0
            batch_time = time.time()

        if valid_loader != None and i % args.eval_step == 0 and i > 0:
            model.eval()
            _, _ = eval_step(   model       = model,
                                data_loader = valid_loader,
                                criterion   = criterion,
                                step        = epoch * total_step,
                                datasplit   = valid_type)

            model.train()


def eval_step( model, data_loader,  criterion, step, datasplit):
    model.eval()
    total_step      = len(data_loader)
    start_time      = time.time()
    epoch_loss      = 0.
    results_dict    = vp_dict()

    for i, (images, label) in enumerate(data_loader):

        if i % args.log_rate == 0:
            print("Evaluation of %s [%d/%d] Time Elapsed: %f " % (datasplit, i, total_step, time.time() - start_time))

        images  = to_var(images, volatile=True)
        label   = to_var(label, volatile=True)

        preds   = model(images)

        epoch_loss+= criterion(preds, label)


    logger.add_scalar_value("(" + args.dataset + ") Accuracy",     accuracy ,step=step)

    epoch_loss = float(epoch_loss)
    assert type(epoch_loss) == float, 'Error: Loss type is not float'
    return epoch_loss, w_acc


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
    parser.add_argument('--save_epoch',      type=int , default=10)
    parser.add_argument('--eval_epoch',      type=int , default=1)
    parser.add_argument('--eval_step',       type=int , default=1000)
    parser.add_argument('--log_rate',        type=int, default=10)
    parser.add_argument('--num_workers',     type=int, default=7)

    # training parameters
    parser.add_argument('--num_epochs',         type=int,   default=100)
    parser.add_argument('--batch_size',         type=int,   default=256)
    parser.add_argument('--lr',                 type=float, default=0.01)
    parser.add_argument('--optimizer',          type=str,   default='sgd')
    parser.add_argument('--annealing_factor',   type=float, default=0.988)

    # experiment details
    parser.add_argument('--dataset',         type=str, default='pascalKP')
    parser.add_argument('--model',           type=str, default='pretrained_clickhere')
    parser.add_argument('--experiment_name', type=str, default= 'Test')
    parser.add_argument('--evaluate_only',   action="store_true",default=False)
    parser.add_argument('--evaluate_train',  action="store_true",default=False)
    parser.add_argument('--resume',           type=str, default=None)

    # Magnet Loss Parameters
    parser.add_argument('--M',      type=int, default=12)       # Number of nearest clusters per mini-batch
    parser.add_argument('--K',      type=int, default=12)       # Number of clusters per class
    parser.add_argument('--D',      type=int, default=12)       # Number of examples per cluster


    args = parser.parse_args()


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
    print(args)

    # Define Logger
    tensorboard_logdir = '/z/home/mbanani/tensorboard_logs'
    log_name    = args.full_experiment_name
    logger      = Logger(os.path.join(tensorboard_logdir, log_name))

    main(args)
