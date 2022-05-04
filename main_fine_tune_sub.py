"""
use for CNN_basis.py
test the acc of 9 subject
cross validation data
"""

from preprocess import *
# from vgg_for_csp import cnn
import matplotlib.pyplot as plt
# from train import train
from train_fine_tune_sub import train
import os

subjects=[i+1 for i in range(13)]
subjects.remove(9)
# sub_index = 2
for sub_index in range(9):
    # sub_index += 1
    # sub_index = 9
    A_data_train, A_label_train, A_data_test, A_label_test = split_subject(1,'A')
    B_data_train, B_label_train, B_data_test, B_label_test = split_subject(sub_index + 1, 'A')


    train(sub_index+1, A_data_train, A_label_train, B_data_train, B_label_train, B_data_test, B_label_test,50,[0],'pyramidViT_fine_tune_sub')




