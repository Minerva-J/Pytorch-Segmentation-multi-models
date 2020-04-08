# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt


def encode_mask(mask, labels):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(labels):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_mask(label_mask, labels, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if isinstance(label_mask, torch.Tensor):
        label_mask = label_mask.cpu().numpy()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, len(labels)):
        r[label_mask == ll] = labels[ll, 0]
        g[label_mask == ll] = labels[ll, 1]
        b[label_mask == ll] = labels[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def decode_mask_seq(label_masks, labels):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_masks.append(decode_mask(label_mask, labels))
    return torch.from_numpy(np.array(rgb_masks).transpose(0, 3, 1, 2))
