import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import segmentation_models as sm
from segmentation_models.utils import set_trainable
from dataset import DataGenerator



if __name__ == '__main__':
    # hyperparameter
    image_size = 384
    train_path = '../data/train/imgs/'  # address of the dataset
    mask_path = '../data/train/masks/'
    epochs = 200  # number of time we need to train dataset
    lr = 1e-4
    batch_size = 4  # tarining batch size
    val_rate = 0.2

    # train path
    train_ids = os.listdir(train_path)
    print(len(train_ids))
    # Validation Data Size
    n_val = int(len(train_ids) * val_rate)  # size of validation set


    valid_ids = train_ids[:n_val]  # list of image ids used for validation of result 0 to 9
    train_ids = train_ids[n_val:]  # list of image ids used for training dataset
    # print(valid_ids, "\n\n")
    print("training_size: ", len(train_ids), "validation_size: ", len(valid_ids))

    train_gen = DataGenerator(train_ids, train_path, mask_path, img_size=image_size, batch_size=batch_size)
    valid_gen = DataGenerator(valid_ids, train_path, mask_path, img_size=image_size, batch_size=batch_size)

    print("total training batches: ", len(train_gen))
    print("total validaton batches: ", len(valid_gen))
    train_steps = len(train_ids) // batch_size
    valid_steps = len(valid_ids) // batch_size

    BACKBONE = 'resnet50'
    #preprocess_input = sm.get_preprocessing(BACKBONE)

    # define model
    model = sm.Unet(BACKBONE, classes=1, encoder_weights='imagenet')
    
    optimizer = optimizers.Adam(lr=lr, decay=1e-4)
    model.compile(
        optimizer=optimizer,
#        "Adam",
        loss=sm.losses.bce_dice_loss, # sm.losses.bce_jaccard_loss, # sm.losses.binary_crossentropy,
        metrics=[sm.metrics.iou_score],
    )
    #model.summary()

    callbacks = [
        EarlyStopping(patience=15, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-7, verbose=1),
        ModelCheckpoint('./U_checkpoints/weights.Epoch{epoch:03d}-Loss{loss:.3f}-VIou{val_iou_score:.3f}.h5', verbose=1,
                        monitor='val_iou_score', mode="max", save_best_only=True, save_weights_only=True)]

    # fit model
    set_trainable(model)
    model.fit_generator(generator=train_gen, validation_data=valid_gen,
                        steps_per_epoch=train_steps, validation_steps=valid_steps,
                        epochs=epochs, callbacks=callbacks)

"""
hist = model.fit_gen...
확인해보면 modelcheckpoint에 저장할 수 있는 로그들, epoch, loss, metirc등의
key를 확인해볼 수 있다.
    for key in hist.history:
        print(key)
"""