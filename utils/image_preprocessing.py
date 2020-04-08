# -*- coding: utf-8 -*-
__author__ = '106360'
import numpy as np
import skimage, skimage.io
import skimage.color
import cv2

import skimage.morphology


def estimate_dc(imgar, ps=20):
    se = np.ones((ps, ps), dtype=bool)
    minIm = np.minimum(imgar[:, :, 0], imgar[:, :, 1])
    
    minIm = np.minimum(minIm, imgar[:, :, 2])
    return skimage.morphology.erosion(minIm, se)


def estimate_atm(imgar, jdark, px=1e-3):
    """
    Automatic atmospheric light estimation. According to section (4.4) in the reference paper
    http://kaiminghe.com/cvpr09/index.html

    Parameters
    -----------
    imgar:    an H*W RGB hazed image
    jdark:    the dark channel of imgar
    px:       the percentage of brigther pixels to be considered (default=1e-3, i.e. 0.1%)

    Return
    -----------
    The atmosphere light estimated in imgar, A (a RGB vector).
    """

    # reshape both matrix to get it in 1-D array shape
    imgavec = np.resize(imgar, (imgar.shape[0] * imgar.shape[1], imgar.shape[2]))
    jdarkvec = np.reshape(jdark, jdark.size)

    # the number of pixels to be considered
    numpx = np.int(jdark.size * px)

    # index sort the jdark channel in descending order
    isjd = np.argsort(-jdarkvec)

    asum = np.array([0.0, 0.0, 0.0])
    for i in range(0, numpx):
        asum[:] += imgavec[isjd[i], :]

    A = np.array([0.0, 0.0, 0.0])
    A[:] = asum[:] / numpx

    # returns the calculated airlight A
    return A


def estimate_tr(imgar, A, w=0.75):
    """
    Transmission estimation. According to section (4.1) equation (11) in the reference paper
    http://kaiminghe.com/cvpr09/index.html

    Parameters
    -----------
    imgar:    an H*W RGB hazed image
    A:        the atmospheric light of imgar
    w:        the omega weight parameter, the amount of haze to be removed (default=0.95)

    Return
    -----------
    The transmission estimated in imgar, t (a H*W matrix).
    """
    # the normalized haze image
    nimg = imgar

    # calculate the normalized haze image
    for c in range(0, imgar.shape[2]):
        nimg[:, :, c] = imgar[:, :, c] / A[c]

    # estimate the dark channel of the normalized haze image
    njdark = estimate_dc(nimg)

    # calculates the transmisson t
    t = 1 - w * njdark

    return t


def boxfilter(img, r):
    (rows, cols) = img.shape
    imDst = np.zeros_like(img)

    imCum = np.cumsum(img, 0)
    imDst[0: r + 1, :] = imCum[r: 2 * r + 1, :]
    imDst[r + 1: rows - r, :] = imCum[2 * r + 1: rows, :] - imCum[0: rows - 2 * r - 1, :]
    imDst[rows - r: rows, :] = np.tile(imCum[rows - 1, :], [r, 1]) - imCum[rows - 2 * r - 1: rows - r - 1, :]

    imCum = np.cumsum(imDst, 1)
    imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
    imDst[:, r + 1: cols - r] = imCum[:, 2 * r + 1: cols] - imCum[:, 0: cols - 2 * r - 1]
    imDst[:, cols - r: cols] = np.tile(imCum[:, cols - 1], [r, 1]).T - imCum[:, cols - 2 * r - 1: cols - r - 1]

    return imDst


def guidedfilter(I, p, r, eps):
    (rows, cols) = I.shape
    N = boxfilter(np.ones([rows, cols]), r)

    meanI = boxfilter(I, r) / N
    meanP = boxfilter(p, r) / N
    meanIp = boxfilter(I * p, r) / N
    covIp = meanIp - meanI * meanP

    meanII = boxfilter(I * I, r) / N
    varI = meanII - meanI * meanI

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = boxfilter(a, r) / N
    meanB = boxfilter(b, r) / N

    q = meanA * I + meanB
    return q


def invert_model(I, ref_transmission, airlight):
    restored_image = np.zeros_like(I)
    M = restored_image.shape[0]
    N = restored_image.shape[1]
    K = restored_image.shape[2]

    for i in range(K):
        restored_image[:, :, i] = (I[:, :, i] - airlight[i] * np.ones((M, N))) / np.maximum(ref_transmission, 0.1) + \
                                  airlight[i]

    return np.clip(restored_image, 0, 1)


def correct_illumination(RGBImage, radius=20, r_guid=20, eps_guid=0.1):
    I = skimage.img_as_float(RGBImage)
    I = 1 - I;
    
    # Build Dark Channel Image
    dC = estimate_dc(I, radius)

    # plt.imshow(dC)
    # plt.show()

    # Compute Airlight
    airlight = estimate_atm(I, dC);#A

    # Rough estimate of the transmission map

    transmission = estimate_tr(I, airlight)#t(x)

    # plt.imshow(transmission)
    # plt.show()

    # Refine Transmission with scalar(luminance) guided filter

    im_v = skimage.color.rgb2gray(I);
    # im_v = im_hsv[:, :, 2];

    ref_transmission = guidedfilter(im_v, transmission, r_guid, eps_guid)

    # plt.imshow(ref_transmission)
    # plt.show()

    # Invert the haze formation model

    restored = invert_model(I, ref_transmission, airlight)

    # plt.imshow(restored)
    # plt.show()

    # Undo intensity inversion
    restored = 1 - restored;

    restored = np.minimum(restored, 1.0)
    restored = np.maximum(restored, 0.0)
    return restored

#### COLOR TRANSFER
def image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)

    (lMean, lStd) = (l.mean(), l.std())
    (lMean2, lStd2) = (np.mean(l[np.where(l > 1)]), np.std(l[np.where(l > 1)]))

    (aMean, aStd) = (a.mean(), a.std())
    (aMean2, aStd2) = (np.mean(a[np.where(l > 1)]), np.std(a[np.where(l > 1)]))

    (bMean, bStd) = (b.mean(), b.std())
    (bMean2, bStd2) = (np.mean(b[np.where(l > 1)]), np.std(b[np.where(l > 1)]))

    #     # return the color statistics
    return (lMean, lMean2, aMean, aMean2, bMean, bMean2)


def color_transfer(source, target):
    source = np.array(source)[:, :, ::-1]
    target = np.array(target)[:, :, ::-1]
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # scale by the standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer[:, :, ::-1]