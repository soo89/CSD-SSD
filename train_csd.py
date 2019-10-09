from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
# from ssd_consistency import build_ssd_con
from csd import build_ssd_con
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import math
import copy


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC300', choices=['VOC300', 'VOC512'],
                    type=str, help='VOC300 or VOC512')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,  # None  'weights/ssd300_COCO_80000.pth'
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC300':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc300
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    elif args.dataset == 'VOC512':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc512
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    finish_flag = True

    while(finish_flag):
        ssd_net = build_ssd_con('train', cfg['min_dim'], cfg['num_classes'])
        net = ssd_net

        if args.cuda:
            net = torch.nn.DataParallel(ssd_net)
            cudnn.benchmark = True

        if args.resume:
            print('Resuming training, loading {}...'.format(args.resume))
            ssd_net.load_weights(args.resume)
        else:
            vgg_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network...')
            ssd_net.vgg.load_state_dict(vgg_weights)

        if args.cuda:
            net = net.cuda()

        if not args.resume:
            print('Initializing weights...')
            # initialize newly added layers' weights with xavier method
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)

        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                 False, args.cuda)
        conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()


        net.train()
        # loss counters
        loc_loss = 0
        conf_loss = 0
        epoch = 0
        supervised_flag = 1
        print('Loading the dataset...')

        step_index = 0


        if args.visdom:
            vis_title = 'SSD.PyTorch on ' + dataset.name
            vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
            iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
            epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)


        total_un_iter_num = 0


        supervised_batch =  args.batch_size
        #unsupervised_batch = args.batch_size - supervised_batch
        #data_shuffle = 0

        if(args.start_iter==0):
            supervised_dataset = VOCDetection_con_init(root=args.dataset_root,
                                                     transform=SSDAugmentation(cfg['min_dim'],
                                                                                  MEANS))
        else:
            supervised_flag = 0
            supervised_dataset = VOCDetection_con(root=args.dataset_root,
                                                     transform=SSDAugmentation(cfg['min_dim'],
                                                                                  MEANS))#,shuffle_flag=data_shuffle)
            #data_shuffle = 1

        supervised_data_loader = data.DataLoader(supervised_dataset, supervised_batch,
                                                   num_workers=args.num_workers,
                                                   shuffle=True, collate_fn=detection_collate,
                                                   pin_memory=True, drop_last=True)


        batch_iterator = iter(supervised_data_loader)

        for iteration in range(args.start_iter, cfg['max_iter']):
            if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
                update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                                'append', epoch_size)
                # reset epoch loss counters
                loc_loss = 0
                conf_loss = 0
                epoch += 1

            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            try:
                images, targets, semis = next(batch_iterator)
            except StopIteration:
                supervised_flag = 0
                supervised_dataset = VOCDetection_con(root=args.dataset_root,
                                                         transform=SSDAugmentation(cfg['min_dim'],
                                                                                      MEANS))#, shuffle_flag=data_shuffle)
                supervised_data_loader = data.DataLoader(supervised_dataset, supervised_batch,
                                                           num_workers=args.num_workers,
                                                           shuffle=True, collate_fn=detection_collate,
                                                           pin_memory=True, drop_last=True)
                batch_iterator = iter(supervised_data_loader)
                images, targets, semis = next(batch_iterator)


            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
            # forward
            t0 = time.time()

            out, conf, conf_flip, loc, loc_flip = net(images)


            sup_image_binary_index = np.zeros([len(semis),1])

            for super_image in range(len(semis)):
                if(int(semis[super_image])==1):
                    sup_image_binary_index[super_image] = 1
                else:
                    sup_image_binary_index[super_image] = 0

                if(int(semis[len(semis)-1-super_image])==0):
                    del targets[len(semis)-1-super_image]


            sup_image_index = np.where(sup_image_binary_index == 1)[0]
            unsup_image_index = np.where(sup_image_binary_index == 0)[0]

            loc_data, conf_data, priors = out

            if (len(sup_image_index) != 0):
                loc_data = loc_data[sup_image_index,:,:]
                conf_data = conf_data[sup_image_index,:,:]
                output = (
                    loc_data,
                    conf_data,
                    priors
                )

            # backprop
            # loss = Variable(torch.cuda.FloatTensor([0]))
            loss_l = Variable(torch.cuda.FloatTensor([0]))
            loss_c = Variable(torch.cuda.FloatTensor([0]))



            if(len(sup_image_index)!=0):
                try:
                    loss_l, loss_c = criterion(output, targets)
                except:
                    break
                    print('--------------')


            sampling = True
            if(sampling is True):
                conf_class = conf[:,:,1:].clone()
                background_score = conf[:, :, 0].clone()
                each_val, each_index = torch.max(conf_class, dim=2)
                mask_val = each_val > background_score
                mask_val = mask_val.data

                mask_conf_index = mask_val.unsqueeze(2).expand_as(conf)
                mask_loc_index = mask_val.unsqueeze(2).expand_as(loc)

                conf_mask_sample = conf.clone()
                loc_mask_sample = loc.clone()
                conf_sampled = conf_mask_sample[mask_conf_index].view(-1, 21)
                loc_sampled = loc_mask_sample[mask_loc_index].view(-1, 4)

                conf_mask_sample_flip = conf_flip.clone()
                loc_mask_sample_flip = loc_flip.clone()
                conf_sampled_flip = conf_mask_sample_flip[mask_conf_index].view(-1, 21)
                loc_sampled_flip = loc_mask_sample_flip[mask_loc_index].view(-1, 4)

            if(mask_val.sum()>0):
                ## JSD !!!!!1
                conf_sampled_flip = conf_sampled_flip + 1e-7
                conf_sampled = conf_sampled + 1e-7
                consistency_conf_loss_a = conf_consistency_criterion(conf_sampled.log(), conf_sampled_flip.detach()).sum(-1).mean()
                consistency_conf_loss_b = conf_consistency_criterion(conf_sampled_flip.log(), conf_sampled.detach()).sum(-1).mean()
                consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b

                ## LOC LOSS
                consistency_loc_loss_x = torch.mean(torch.pow(loc_sampled[:, 0] + loc_sampled_flip[:, 0], exponent=2))
                consistency_loc_loss_y = torch.mean(torch.pow(loc_sampled[:, 1] - loc_sampled_flip[:, 1], exponent=2))
                consistency_loc_loss_w = torch.mean(torch.pow(loc_sampled[:, 2] - loc_sampled_flip[:, 2], exponent=2))
                consistency_loc_loss_h = torch.mean(torch.pow(loc_sampled[:, 3] - loc_sampled_flip[:, 3], exponent=2))

                consistency_loc_loss = torch.div(
                    consistency_loc_loss_x + consistency_loc_loss_y + consistency_loc_loss_w + consistency_loc_loss_h,
                    4)

            else:
                consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
                consistency_loc_loss = Variable(torch.cuda.FloatTensor([0]))

            consistency_loss = torch.div(consistency_conf_loss,2) + consistency_loc_loss

            ramp_weight = rampweight(iteration)
            consistency_loss = torch.mul(consistency_loss, ramp_weight)


            if(supervised_flag ==1):
                loss = loss_l + loss_c + consistency_loss
            else:
                if(len(sup_image_index)==0):
                    loss = consistency_loss
                else:
                    loss = loss_l + loss_c + consistency_loss


            if(loss.data>0):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t1 = time.time()
            if(len(sup_image_index)==0):
                loss_l.data = Variable(torch.cuda.FloatTensor([0]))
                loss_c.data = Variable(torch.cuda.FloatTensor([0]))
            else:
                loc_loss += loss_l.data  # [0]
                conf_loss += loss_c.data  # [0]


            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f || consistency_loss : %.4f ||' % (loss.data, consistency_loss.data), end=' ')
                print('loss: %.4f , loss_c: %.4f , loss_l: %.4f , loss_con: %.4f, lr : %.4f, super_len : %d\n' % (loss.data, loss_c.data, loss_l.data, consistency_loss.data,float(optimizer.param_groups[0]['lr']),len(sup_image_index)))


            if(float(loss)>100):
                break

            if args.visdom:
                update_vis_plot(iteration, loss_l.data, loss_c.data,
                                iter_plot, epoch_plot, 'append')

            if iteration != 0 and (iteration+1) % 40000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                           repr(iteration+1) + '.pth')
        # torch.save(ssd_net.state_dict(), args.save_folder + '' + args.dataset + '.pth')
        print('-------------------------------\n')
        print(loss.data)
        print('-------------------------------')

        if((iteration +1) ==cfg['max_iter']):
            finish_flag = False


def rampweight(iteration):
    ramp_up_end = 32000
    ramp_down_start = 100000

    if(iteration<ramp_up_end):
        ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end),2))
    elif(iteration>ramp_down_start):
        ramp_weight = math.exp(-12.5 * math.pow((1 - (120000 - iteration) / 20000),2)) 
    else:
        ramp_weight = 1 


    if(iteration==0):
        ramp_weight = 0

    return ramp_weight




def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
