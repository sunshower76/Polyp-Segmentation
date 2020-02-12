import argparse
import logging
import os
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
import skimage.transform as transform

from dice_loss import binaryDiceCoeff


def predict_img(net,
                full_img,
                device,
                resizing=572,
                out_threshold=0.5):

    net.eval() # evaluation mode. parameters aren't updated

    img = torch.from_numpy(BasicDataset.preprocess(full_img, resizing))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='./checkpoints/CP_epoch40.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', default="../data/test1/imgs/ETIS_imgs_png")
    #"./data/test2/imgs/CVC_Original" , "./data/test1/imgs/ETIS_imgs_png"
    #"./data/test2/masks/CVC_Ground Truth"./data/test1/masks/ETIS_Ground Truth_png"
    parser.add_argument('--target', '-tar', metavar='TARGET', nargs='+',
                        help='filenames of input masks', default="../data/test1/masks/ETIS_Ground Truth_png")

    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', default="../data/result/test1",
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--resizing', '-s', type=float,
                        help="Scale factor for the input images",
                        default=384)
    parser.add_argument('--residual', '-m1', dest='residual', type=bool, default=True,
                        help='add residual connection')
    parser.add_argument('--cbam', '-m2', dest='cbam', type=bool, default=False,
                        help='add CBAM')
    parser.add_argument('--aspp', '-m3', dest='aspp', type=bool, default=True,
                        help='add ASPP')

    return parser.parse_args()


def get_output_filenames(args):
    in_files = os.listdir(args.input)
    in_files.sort()

    out_files = []

    for f in in_files:
        pathsplit = os.path.splitext(f)
        out_files.append(args.output + "./{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    if len(in_files) != len(out_files):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()

    return out_files


def mask_to_image(mask):
    return (mask * 255).astype(np.uint8)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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

    extra_parts = [args.residual, args.cbam, args.aspp]
    net = UNet(n_channels=3, n_classes=1, extra_parts=extra_parts)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if torch.cuda.device_count() > 1:
        #net = nn.DataParallel(net)
        net.to(device)
    else:
        net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    total_num = len(in_files)
    sum_dice = 0.0
    for i, fn in enumerate(in_files):
        logging.info("\nPredicting +image {} ...".format(fn))

        img = io.imread(fn)

        mask = predict_img(net=net,
                           full_img=img,
                           resizing=args.resizing,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            cv2.imwrite(out_files[i], result)

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)

        target = io.imread(target_files[i])
        target = transform.resize(target, (args.resizing, args.resizing), anti_aliasing=True)

        mask = np.expand_dims(mask, 0)  # for calculate dice coefficient
        target = np.expand_dims(target, 0)   # for calculate dice coefficient
        mask = torch.from_numpy(mask).to(device='cuda', dtype=torch.float32)
        target = torch.from_numpy(target).to(device='cuda', dtype=torch.float32)

        dice_score = binaryDiceCoeff(mask, target)
        sum_dice += dice_score
        print("Img : {0} - dice coeff : {1}".format(fn.split("/")[-1], dice_score))

    print("Total average dice score : {}".format(sum_dice/total_num))