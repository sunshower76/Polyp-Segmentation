import os
"""
# If you have multi-gpu, designate the number of GPU to use.
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
"""

import argparse
import logging

from tqdm import tqdm # progress bar
import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import segmentation_models as sm
from segmentation_models.utils import set_trainable
from dataset import DataGenerator

def train_model(model, train_gen, valid_gen, epochs, batch_size, save_cp=True):
    total_batch_count = 0
    train_batch_num = len(train_gen)
    train_num = train_batch_num * batch_size
    #train_gen_out = iter_sequence_infinite(train_gen)

    valid_batch_num = len(valid_gen)
    valid_num = valid_batch_num * batch_size
    #valid_gen_out = iter_sequence_infinite(valid_gen)

    for epoch in range(epochs): # interation as many epochs
        set_trainable(model)
        
        epoch_loss = 0 # loss in this epoch
        epoch_iou = 0
        count = 0

        with tqdm(total=train_num, desc=f'Epoch {epoch + 1}/{epochs}',  position=0, leave=True, unit='img') as pbar:  # make progress bar
            for batch in train_gen:
                #batch = next(train_gen_out)
                imgs = batch[0]
                true_masks = batch[1]
                loss, iou = model.train_on_batch(imgs, true_masks)  # value of loss of this batch
                epoch_loss += loss
                epoch_iou += iou

                pbar.set_postfix(**{'Batch loss': loss, 'Batch IoU': iou})  # floating the loss at the post in the pbar

                pbar.update(imgs.shape[0])  # update progress
                count += 1
                total_batch_count += 1

        print( "Epoch : loss: {}, IoU : {}".format(epoch_loss/count, epoch_iou/count))

        # Do validation
        validation_model(model, valid_gen, valid_num)
        train_gen.on_epoch_end()
        valid_gen.on_epoch_end()

        if save_cp:
            try:
                if not os.path.isdir(checkpoint_dir):
                    os.mkdir(checkpoint_dir)
                    logging.info('Created checkpoint directory')
                else:
                    pass
            except OSError:
                pass
            model.save_weights(os.path.join(checkpoint_dir , f'CP_epoch{epoch + 1}.h5'))
            logging.info(f'Checkpoint {epoch + 1} saved !')

def validation_model(model, valid_gen, valid_num):
    epoch_loss = 0  # loss in this epoch
    epoch_iou = 0
    count = 0

    with tqdm(total=valid_num, desc='Validation round',  position=0, leave=True, unit='img') as pbar:  # make progress bar
        for batch in valid_gen:
            #batch = next(valid_gen_out)
            imgs = batch[0]
            true_masks = batch[1]
            loss, iou = model.test_on_batch(imgs, true_masks)  # value of loss of this batch
            epoch_loss += loss
            epoch_iou += iou

            pbar.set_postfix(**{'Batch, loss': loss, 'Batch IoU': iou})  # floating the loss at the post in the pbar

            pbar.update(imgs.shape[0])  # update progress
            count += 1

    print("Validation loss: {}, IoU: {}".format(epoch_loss / count, epoch_iou / count))
    pred_mask = model.predict(np.expand_dims(imgs[0],0))
    plt.subplot(131)
    plt.imshow(imgs[0])
    plt.subplot(132)
    plt.imshow(true_masks[0].squeeze(), cmap="gray")
    plt.subplot(133)
    plt.imshow(pred_mask.squeeze(), cmap="gray")
    plt.show()
    print()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-bb', '--backbone', default='resnet50', metavar='FILE',
                        help="backcone name")
    parser.add_argument('-w', '--weight', dest='load', type=str, default=False,
                        help='Load model from a .h5 file')
    parser.add_argument('-s', '--resizing', dest='resizing', type=int, default=384,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    img_dir = '../data/train/imgs/'  # ./data/train/imgs/CVC_Original/'
    mask_dir = '../data/train/masks/'  # ./data/train/masks/CVC_Ground Truth/'
    checkpoint_dir = './checkpoints'
    args = get_args()

    # train path
    train_ids = os.listdir(img_dir)
    # Validation Data Size
    n_val = int(len(train_ids) * args.val/100)  # size of validation set


    valid_ids = train_ids[:n_val]  # list of image ids used for validation of result 0 to 9
    train_ids = train_ids[n_val:]  # list of image ids used for training dataset
    # print(valid_ids, "\n\n")
    print("training_size: ", len(train_ids), "validation_size: ", len(valid_ids))

    train_gen = DataGenerator(train_ids, img_dir, mask_dir, img_size=args.resizing, batch_size=args.batch_size)
    valid_gen = DataGenerator(valid_ids, img_dir, mask_dir, img_size=args.resizing, batch_size=args.batch_size)

    print("total training batches: ", len(train_gen))
    print("total validaton batches: ", len(valid_gen))
    train_steps = len(train_ids) // args.batch_size
    valid_steps = len(valid_ids) // args.batch_size

    # define model
    model = sm.Unet(args.backbone, encoder_weights='imagenet')

    optimizer = optimizers.Adam(lr=args.lr, decay=1e-4)
    model.compile(
        optimizer=optimizer,
        #        "Adam",
        loss=sm.losses.bce_dice_loss,  # sm.losses.bce_jaccard_loss, # sm.losses.binary_crossentropy,
        metrics=[sm.metrics.iou_score],
    )
    #model.summary()

    callbacks = [
        EarlyStopping(patience=6, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint('./weights.Epoch{epoch:02d}-Loss{loss:.3f}-VIou{val_iou_score:.3f}.h5', verbose=1,
                        monitor='val_accuracy', save_best_only=True, save_weights_only=True)
                ]


    train_model(model=model, train_gen=train_gen,
                valid_gen=valid_gen, epochs=args.epochs, batch_size=args.batch_size)

