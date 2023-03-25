import numpy as np
import torch
import os

from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tool import pyutils
import argparse
import importlib
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from acdc.dataset import BaseDataSets, RandomGenerator
import random
from torch.nn.modules.loss import CrossEntropyLoss
from val import test_single_volume
from utils import losses
import torch.backends.cudnn as cudnn
from torch.nn.functional import one_hot
import re
from time import strftime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=100, type=int)
    parser.add_argument("--network", default="network.scribformer", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--session_name", default="TransCAM", type=str)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--weights", default='', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--root_path', type=str,
                        default='../data/ACDC', help='Name of Experiment')
    parser.add_argument('--patch_size', type=list, default=[256, 256],
                        help='patch size of network input')
    parser.add_argument('--fold', type=str,
                        default='MAAGfold ', help='cross validation')
    parser.add_argument('--sup_type', type=str,
                        default='scribble', help='supervision type')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--exp', type=str,
                        default='ACDC/scribformer', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='scribformer', help='model_name')
    parser.add_argument('--linear_layer', action="store_true", help='linear layer')
    parser.add_argument('--bilinear', action="store_false", help='use ConvTranspose2d instead of bilinear in Upsample layer')
    parser.add_argument('--weight_cam', type=float, default=0, help='weights of CAM loss')
    parser.add_argument('--weight_cam_subloss', type=float, nargs='+', default=[1, 1, 1, 1], help='sub-weights of CAM loss')

    args = parser.parse_args()
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logdir = os.path.join(snapshot_path, "{}_log.txt".format(strftime("%Y_%m_%d_%H_%M_%S")))
    pyutils.Logger(logdir)
    print("log in ", logdir)
    print(vars(args))

    model = getattr(importlib.import_module(args.network), 'ScribFormer')(linear_layer=args.linear_layer, bilinear=args.bilinear)
    print('model is from', model.__class__)

    tblogger = SummaryWriter(args.tblog_dir)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]), fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path, fold=args.fold, split="val")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    batch_size = args.batch_size
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    num_classes = 4
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)
    cam_loss = CrossEntropyLoss()

    best_performance = 0.0
    best_epoch = 0
    iter_num = 0
    max_iterations = args.max_epoches * len(trainloader)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wt_dec, eps=1e-8)

    model = model.cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    for ep in range(args.max_epoches):
        for iter, sampled_batch in enumerate(trainloader):
            img, label = sampled_batch['image'], sampled_batch['label']
            img, label = img.cuda(), label.cuda()

            pred1, pred2, cam = model(img)

            outputs_soft1 = torch.softmax(pred1, dim=1)
            outputs_soft2 = torch.softmax(pred2, dim=1)

            loss_ce1 = ce_loss(pred1, label[:].cuda().long())
            loss_ce2 = ce_loss(pred2, label[:].cuda().long())
            loss_ce = 0.5 * (loss_ce1 + loss_ce2)

            beta = random.random() + 1e-10
            pseudo_supervision = torch.argmax((beta * outputs_soft1.detach() +
                                               (1.0 - beta) * outputs_soft2.detach()),
                                              dim=1, keepdim=False)
            loss_pse_sup = 0.5 * (dice_loss(outputs_soft1, pseudo_supervision.unsqueeze(1)) +
                                  dice_loss(outputs_soft2, pseudo_supervision.unsqueeze(1)))

            loss = loss_ce + 0.5 * loss_pse_sup

            if args.weight_cam != 0:
                ensemble_pred = (beta * outputs_soft1 + (1.0 - beta) * outputs_soft2)
                for i in range(len(cam)):
                    cam[i] = torch.sigmoid(cam[i])
                weight_cam_subloss = args.weight_cam_subloss
                loss_cam = (0.25 * cam_loss(cam[0], cam[4]) + 0.5 * cam_loss(cam[1], cam[4]) + 0.75 * cam_loss(cam[2], cam[4]) + cam_loss(cam[3], cam[4])) * 0.25
                loss = loss + args.weight_cam * loss_cam

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({'loss': loss.item()})
        else:
            print('epoch: %5d' % ep,
                  'loss: %.4f' % avg_meter.get('loss'), flush=True)

            model.eval()
            metric_list = 0.0
            for i_batch, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume(
                    sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                metric_list += np.array(metric_i)
            metric_list = metric_list / len(db_val)
            for class_i in range(num_classes - 1):
                tblogger.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                  metric_list[class_i, 0], ep)
                tblogger.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                  metric_list[class_i, 1], ep)

            performance = np.mean(metric_list, axis=0)[0]

            mean_hd95 = np.mean(metric_list, axis=0)[1]
            tblogger.add_scalar('info/val_mean_dice', performance, ep)
            tblogger.add_scalar('info/val_mean_hd95', mean_hd95, ep)

            if performance > 0.85:
                print("Update high dice score model!")
                file_name = os.path.join(snapshot_path, '{}_{}_model.pth'.format(args.model, str(performance)[0:6]))
                torch.save(model.state_dict(), file_name)
            if (ep+1) % 100 == 0:
                print("{} model!".format(ep))
                file_name = os.path.join(snapshot_path, '{}_{}_model.pth'.format(args.model, ep))
                torch.save(model.state_dict(), file_name)
            if performance > best_performance:
                best_performance = performance
                best_epoch = ep

                save_best = os.path.join(snapshot_path,
                                         '{}_best_model.pth'.format(args.model))

                torch.save(model.state_dict(), save_best)
                print('best model in epoch %5d  mean_dice : %.4f' % (ep, performance))

            print(
                'epoch %5d  mean_dice : %.4f mean_hd95 : %.4f' % (ep, performance, mean_hd95), flush=True)
            model.train()

            avg_meter.pop()
    print('best model in epoch %5d  mean_dice : %.4f' % (best_epoch, best_performance))
    print('save best model in {}/{}_best_model.pth'.format(snapshot_path, args.model))
    torch.save(model.state_dict(), os.path.join(snapshot_path,
                                                           '{}_final_model.pth'.format(args.model)))
