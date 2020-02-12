import os
"""
# If you have multi-gpu, designate the number of GPU to use.
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
"""

import argparse
import logging
import cv2
from skimage import io
import skimage.transform as transform
import numpy as np
import segmentation_models as sm
from eval import get_dice


def predict_img(net,
                full_img,
                resizing=384,
                out_threshold=0.5):

    img = transform.resize(full_img, (resizing,resizing), anti_aliasing=True)
    img = np.expand_dims(img, 0)

    output = net.predict(img)
    output = output.squeeze()

    return output > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='efficientnetb1', # resnet50, densnet169,... possible.
                        metavar='FILE',
                        help="backcone name")
    parser.add_argument('--weight', '-w', default='./EffiB1U_checkpoints/weight_name...',  # put in the 'h5'file name
                        metavar='FILE',
                        help="Specify the file in which the weight stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', default= "../data/test2/imgs/CVC_Original")
    #"../data/test2/imgs/CVC_Original" ,  "../data/test1/imgs/ETIS_imgs_png"
    #"../data/test2/masks/CVC_Ground Truth",  "../data/test1/masks/ETIS_Ground Truth_png"
    parser.add_argument('--target', '-tar', metavar='TARGET', nargs='+',
                        help='filenames of input masks', default= "./data/test2/masks/CVC_Ground Truth")

    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', default="./results/U_Dens169_result/test2",
                        help='Filenames of ouput images')
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--resizing', '-s', type=float,
                        help="Scale factor for the input images",
                        default=384)
    return parser.parse_args()


def get_output_filenames(args):
    in_files = os.listdir(args.input)
    in_files.sort()

    out_files = []

    for f in in_files:
        pathsplit = os.path.splitext(f)
        out_files.append(args.output + "/{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    if len(in_files) != len(out_files):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()

    return out_files


def mask_to_image(mask):
    return (mask * 255).astype(np.uint8)


if __name__ == "__main__":
    args = get_args()

    in_files = list()
    in_files_name = os.listdir(args.input)
    in_files_name.sort()

    target_files = list()
    target_names = os.listdir(args.target)
    target_names.sort()

    for i, in_file in enumerate(in_files_name):
        in_file_path = os.path.join(args.input, in_file)
        in_files.append(in_file_path)

        target_file_path = os.path.join(args.target, target_names[i])
        target_files.append(target_file_path)

    out_files = get_output_filenames(args)

    model = sm.Unet(args.model)
    model.load_weights(args.weight)
    # We need to copile model. but it is not important because we will predict not train. It is formulaic
    model.compile(
        "Adam",
        loss=sm.losses.bce_dice_loss,  # sm.losses.bce_jaccard_loss, # sm.losses.binary_crossentropy,
        metrics=[sm.metrics.iou_score],
    )

    logging.info("Model loaded !")

    total_num = len(in_files)
    sum_dice = 0.0
    for i, fn in enumerate(in_files):
        logging.info("\nPredicting +image {} ...".format(fn))

        img = io.imread(fn)
        mask = predict_img(model, img, resizing=args.resizing)

        if args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            cv2.imwrite(out_files[i], result)

            logging.info("Mask saved to {}".format(out_files[i]))

        target = io.imread(target_files[i])
        target = transform.resize(target, (args.resizing, args.resizing), anti_aliasing=True)

        mask = np.expand_dims(mask, 0)  # for calculate dice coefficient
        target = np.expand_dims(target, 0)   # for calculate dice coefficient

        dice_score = get_dice(mask, target)
        sum_dice += dice_score
        print("Img : {0} - dice coeff : {1}".format(fn.split("/")[-1], dice_score))

    print("Total average dice score : {}".format(sum_dice/total_num))