import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm # progress bar

from eval import eval_net
from unet import UNet
from dice_loss import binaryDiceLoss

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split


dir_img = '../data/train/imgs/'
dir_mask = '../data/train/masks/'
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

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_RESIZING_{resizing}')
    global_step = 0  # number of processed batch up to now (네트워크에서 현재까지 처리된 배치 수)

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

    if net.n_classes > 1:  # loss function : case of multi class
        criterion = nn.CrossEntropyLoss()
    else:  # loss function : case of binary class
        criterion = binaryDiceLoss  # nn.BCEWithLogitsLoss()

    for epoch in range(epochs):  # interation as many epochs
        net.train()

        epoch_loss = 0 # loss in this epoch
        count = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:  # make progress bar
            for batch in train_loader:
                imgs = batch['image']  # from dataloader dictionary
                true_masks = batch['mask']  # from dataloader dictionary
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)  # load to GPU
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)  # load to GPU

                masks_pred = net(imgs)  # forward image to Unet
                loss = criterion(masks_pred, true_masks)  # calculate loss
                epoch_loss += loss.item()  # loss.item() = value of loss of this batch
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})  # floating the loss at the post in the pbar

                optimizer.zero_grad()  # initialize before gradient descent
                loss.backward()  # back-propagation
                optimizer.step()  # update parameters of Unet

                pbar.update(imgs.shape[0])  # update progress
                global_step += 1
                count += 1
                # validation in every

        print( "loss (epoch loss) : {}".format(epoch_loss/count))

        # validation == True !
        val_score = eval_net(net, val_loader, device, n_val)
        logging.info('Validation dice score {}'.format(val_score))

        if save_cp:
            try:
                if not os.path.isdir(dir_checkpoint):
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                else:
                    pass
            except OSError:
                pass
            torch.save(net.state_dict(),
                       os.path.join(dir_checkpoint , f'CP_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--resizing', dest='resizing', type=int, default=384,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')

    # You change True/False. Which extra parts will you insert?
    parser.add_argument('-m1', '--residual', dest='residual', type=bool, default=True,
                        help='add residual connection')
    parser.add_argument('-m2', '--cbam', dest='cbam', type=bool, default=True,
                        help='add CBAM')
    parser.add_argument('-m3', '--aspp', dest='aspp', type=bool, default=True,
                        help='add ASPP')

    return parser.parse_args()


if __name__ == '__main__':

    """
    # If you have multi-gpu, designate the number of GPU to use.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    """

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')
    extra_parts = [args.residual, args.cbam, args.aspp]

    net = UNet(n_channels=3, n_classes=1, extra_parts=extra_parts)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t"Residual Connection : "{args.residual}\n'
                 f'\t"CBAM(attention in contract path) : "{args.cbam}\n'
                 f'\t"ASPP(Atrous Spatial Pyramid Pooling) : "{args.aspp}\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    if torch.cuda.device_count() > 1:
        #net = nn.DataParallel(net)
        net.to(device)
    else:
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
