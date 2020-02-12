import torch
import torch.nn.functional as F
from tqdm import tqdm
from dice_loss import binaryDiceCoeff
import matplotlib.pyplot as plt
import numpy as np

def eval_net(net, loader, device, n_val, is_dice=False):
    print("Validaiton...")
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    cross_tot = 0
    dice_tot = 0
    count = 0
    total_len = len(loader)

    #with tqdm(total=n_val, desc='Validation round', unit='img') as pbar:
    for num, batch in enumerate(loader):
        imgs = batch['image']
        true_masks = batch['mask']

        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_type = torch.float32 if net.n_classes == 1 else torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)

        pred_masks = net(imgs)
        pred_masks = (pred_masks > 0.5).float()
        dice_tot += binaryDiceCoeff(pred_masks, true_masks)

        """
        for true_mask, pred in zip(true_masks, pred_masks):
            pred = (pred > 0.5).float()
            count += 1
            if net.n_classes > 1:
                cross_tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
            else:
                cross_tot += F.binary_cross_entropy_with_logits(pred, true_mask).item()
                dice_tot += binaryDiceCoeff(pred, true_mask)
        """

        if num + 1 >= total_len:
            plt.subplot(131)
            plt.imshow(np.transpose(imgs.cpu().numpy()[0], (1,2,0)))
            plt.subplot(132)
            plt.imshow(true_masks[0].cpu().numpy().squeeze(), cmap="gray")
            plt.subplot(133)
            plt.imshow(pred_masks[0].cpu().detach().numpy().squeeze(), cmap="gray")
            plt.show()
            print()

    return dice_tot/total_len  # cross_tot/total_len  # n_val
