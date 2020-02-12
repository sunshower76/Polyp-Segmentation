import logging
import argparse
import os
import sys

import torch
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm # progress bar

from eval import eval_net
from dice_loss import binaryDiceLoss
from models import resUnet50

from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

#from encoding.nn import DataParallelModel, DataParallelCriterion

dir_img = '../data/train/imgs/'
dir_mask ='../data/train/masks/'
dir_checkpoint = './checkpoints'

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              resizing=572):

    dataset = BasicDataset(dir_img, dir_mask, resizing, is_transform=True)  # img path, mask path, resizing size

    n_val = int(len(dataset) * val_percent)  # size of validation set
    n_train = len(dataset) - n_val  # size of training set
    train, val = random_split(dataset, [n_train, n_val])  # split total dataset to train and validation set with its ratio

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, pin_memory=True)

    global_step = 0  # number of processed batch up to now

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images resizing:  {resizing}
    ''')  # record logs

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4) # SGD optimizer
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', min_lr=1e-7, patience=2, verbose=1)
    criterion = binaryDiceLoss # nn.BCEWithLogitsLoss()
        
    for epoch in range(epochs): # interation as many epochs
        net.train() # at the first epoch and after each validation step.

        epoch_loss = 0 # loss in this epoch
        count = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:  # make progress bar
            for batch in train_loader:
                imgs = batch['image']  # from dataloader dictionary
                true_masks = batch['mask']  # from dataloader dictionary

                # input image channel size must be 3
                assert imgs.shape[1] == 3, \
                    f'Network has been defined with {3} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)  # load to GPU
                mask_type = torch.float32  # if net.n_classes is  1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type) # load to GPU

                pred_masks = net(imgs)  # forward image to Unet
                loss = criterion(pred_masks, true_masks)  # calculate loss
                epoch_loss += loss.item()  # loss.item() = value of loss of this batch

                pbar.set_postfix(**{'loss (each batch)': loss.item()})  # floating the loss at the post in the pbar

                optimizer.zero_grad()  # initialize before gradient descent
                loss.backward()  # back-propagation
                optimizer.step()  # update parameters of Unet

                pbar.update(imgs.shape[0])  # update progress
                global_step += 1
                count += 1
                
        print( "loss (epoch loss) : {}".format(epoch_loss/count))

        # validation == True !
        val_dice_score = eval_net(net, val_loader, device, n_val)
        scheduler.step(epoch_loss)
        logging.info('Validation dice score: {}'.format(val_dice_score))

        if save_cp:
            try:
                if not os.path.isdir(dir_checkpoint):
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                else:
                    pass
            except OSError:
                pass
            torch.save(net.state_dict(), os.path.join(dir_checkpoint , f'CP_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--resizing', dest='resizing', type=int, default=384,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    """
    # If you have multi-gpu, designate the number of GPU to use.
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = resUnet50(n_classes=1, bilinear=True, pretrained=True)
    gpu_num = torch.cuda.device_count() 
    if gpu_num > 1:
        #net = nn.DataParallel(net)
        net.to(device)
        #criterion = DataParallelCriterion(criterion)
    else:
        net.to(device)
    
    logging.info(f'Network:\n'
                 f'\t{3} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  resizing=args.resizing,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
