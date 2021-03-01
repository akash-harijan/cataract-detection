# -*- coding: utf-8 -*-
# import click
import logging
from pathlib import Path
from augment import augment_and_concate
# from dotenv import find_dotenv, load_dotenv



import cv2
import numpy as np
import glob
import os


class DataLoader:
    def __init__(self, data_path, resize_shape=(160, 160)):
        self.data_path = data_path
        self.resized_shape = resize_shape

    def get_data(self, path):

        normal_path = os.path.join(path, 'normal')
        cataract_path = os.path.join(path, 'cataract')

        normal_eyes = glob.glob(normal_path + '/*.jp*')
        cataract_eyes = glob.glob(cataract_path + '/*.jp*')

        n_eyes_imgs = np.array([cv2.resize(cv2.imread(p), self.resized_shape) for p in normal_eyes])
        c_eyes_imgs = np.array([cv2.resize(cv2.imread(p), self.resized_shape) for p in cataract_eyes])

        n_eyes_imgs = np.vstack((n_eyes_imgs, n_eyes_imgs[:, :, ::-1]))
        c_eyes_imgs = np.vstack((c_eyes_imgs, c_eyes_imgs[:, :, ::-1]))

        return n_eyes_imgs, c_eyes_imgs

    def load_data(self):

        test_data_dir = os.path.join(self.data_path, "Test")
        train_data_dir = os.path.join(self.data_path, "Train")

        test_data_cataract, test_data_normal = self.get_data(test_data_dir)
        train_data_cataract, train_data_normal = self.get_data(train_data_dir)

        print("Testing data, Normal : {0} and Cataract : {1}".format(test_data_normal.shape, test_data_cataract.shape))
        print("Training data, Normal : {0} and Cataract : {1}".format(train_data_normal.shape, train_data_cataract.shape))

        x_train, y_train = augment_and_concate(train_data_normal, train_data_cataract)
        x_test, y_test = augment_and_concate(test_data_normal, test_data_cataract)

        return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    print("Project dir {0}".format(project_dir))

    loader = DataLoader(os.path.join(project_dir, "data/external"))
    x_train, x_test, y_train, y_test = loader.load_data()

    print("Xtrain : {0}, X_test : {1}, Y_train : {2}, Y_test : {3}".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

