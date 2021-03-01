import imgaug.augmenters as iaa
import numpy as np


def rotate_imgs(array):
    output = np.zeros_like(array)
    rotate = iaa.Affine(rotate=(-50, 30))
    for i in range(array.shape[0]):
        output[i] = rotate.augment_image(array[i])

    return output


def guassian(array):
    gaussian_noise = iaa.AdditiveGaussianNoise(10, 20)
    output = np.zeros_like(array)
    for i in range(array.shape[0]):
        output[i] = gaussian_noise.augment_image(array[i])

    return output


def sheer(array):
    shear = iaa.Affine(shear=(0, 40))
    output = np.zeros_like(array)
    for i in range(array.shape[0]):
        output[i] = shear.augment_image(array[i])

    return output


def contrast(array):
    contrast_ = iaa.GammaContrast(gamma=2.0)
    output = np.zeros_like(array)
    for i in range(array.shape[0]):
        output[i] = contrast_.augment_image(array[i])

    return output


def scale(array):
    scale_im = iaa.Affine(scale={"x": (1.5, 1.0), "y": (1.5, 1.0)})
    output = np.zeros_like(array)
    for i in range(array.shape[0]):
        output[i] = scale_im.augment_image(array[i])

    return output


def augment_it(array):
    rotated_n_eyes = rotate_imgs(array)
    guassian_n_eyes = guassian(array)
    sheer_n_eyes = sheer(array)
    contrast_n_eyes = contrast(array)
    scale_n_eyes = scale(array)

    return np.vstack((array, rotated_n_eyes, guassian_n_eyes, sheer_n_eyes, contrast_n_eyes, scale_n_eyes))


def augment_and_concate(n_eyes, c_eyes):
    n_eyes_aug = augment_it(n_eyes)
    c_eyes_aug = augment_it(c_eyes)

    y_normal = np.zeros(n_eyes_aug.shape[0])
    y_cataract = np.ones(c_eyes_aug.shape[0])

    x = np.vstack((n_eyes_aug, c_eyes_aug))
    y = np.hstack((y_normal, y_cataract))

    return x, y


