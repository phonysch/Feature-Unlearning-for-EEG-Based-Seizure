# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

import json
from load_data import PrepData
import os
import pickle


def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def save_data(path_name, data_):
    print(f"Saving data to '{path_name}'")
    with open(path_name, 'wb') as file:
        pickle.dump(data_, file)


def load_data(path_name):
    if os.path.isfile(path_name):
        print(f"Loading data from '{path_name}'")
        with open(path_name, 'rb') as file:
            data_ = pickle.load(file)
            return data_
    return None


def my_get_data(remaining_patients, forgetting_patients):

    dataset = 'CHBMIT'
    with open('SETTINGS_%s.json' % dataset) as f:
        settings = json.load(f)

    samples, labels = [], []
    if not os.path.exists('temp/t-sne/data.pth'):
        for index, pat in enumerate(remaining_patients):  # '10' -> 1, 0
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            temp_pre_s = np.transpose(np.concatenate(temp_pre_s).astype(np.float32), (0, 2, 1))
            temp_inter_s = np.transpose(np.concatenate(temp_inter_s).astype(np.float32), (0, 2, 1))

            # # balance the two classes
            if len(temp_inter_s) >= len(temp_pre_s):
                np.random.shuffle(temp_inter_s)
                temp_inter_s = temp_inter_s[:len(temp_pre_s)]
            else:
                np.random.shuffle(temp_pre_s)
                temp_pre_s = temp_pre_s[:len(temp_inter_s)]

            np.random.shuffle(temp_pre_s)
            np.random.shuffle(temp_inter_s)

            temp_pre_s = temp_pre_s[:int(len(temp_pre_s) * 0.02)]
            temp_inter_s = temp_inter_s[:int(len(temp_inter_s) * 0.02)]

            samples.append(temp_pre_s)
            samples.append(temp_inter_s)
            labels.append(np.ones(len(temp_pre_s)).astype(np.int64))
            labels.append(np.ones(len(temp_pre_s)).astype(np.int64) * 2)

        for index, pat in enumerate(forgetting_patients):  # '10' -> 1, 0
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            temp_pre_s = np.transpose(np.concatenate(temp_pre_s).astype(np.float32), (0, 2, 1))
            temp_inter_s = np.transpose(np.concatenate(temp_inter_s).astype(np.float32), (0, 2, 1))

            # # balance the two classes
            if len(temp_inter_s) >= len(temp_pre_s):
                np.random.shuffle(temp_inter_s)
                temp_inter_s = temp_inter_s[:len(temp_pre_s)]
            else:
                np.random.shuffle(temp_pre_s)
                temp_pre_s = temp_pre_s[:len(temp_inter_s)]

            np.random.shuffle(temp_pre_s)
            np.random.shuffle(temp_inter_s)

            temp_pre_s = temp_pre_s[:int(len(temp_pre_s) * 0.05)]
            temp_inter_s = temp_inter_s[:int(len(temp_inter_s) * 0.05)]

            samples.append(temp_pre_s)
            samples.append(temp_inter_s)
            labels.append(np.ones(len(temp_pre_s)).astype(np.int64) * 3)
            labels.append(np.ones(len(temp_pre_s)).astype(np.int64) * 4)

        samples = np.concatenate(samples)
        samples = samples.reshape(-1, samples.shape[1]*samples.shape[2])
        labels = np.concatenate(labels)

        n_samples, n_features = samples.shape[0], samples.shape[1]

        data = [samples, labels]

        save_data('temp/t-sne/data.pth', data)

    else:

        data = load_data('temp/t-sne/data.pth')

        samples, labels = data[0], data[1]
        n_samples, n_features = samples.shape[0], samples.shape[1]

    return samples, labels, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    remaining_patients_ = [
        '1',  #
        '2',  #
        '3',  #
        '5',  #
        '6',  #
        '8',  #
        '9',  #
        '10',  #
        '13',  #
        '14',  #
        '16',  #
        '17',  #
        '18',  #
        '19',  #
        '20',  #
        '21',  #
        '22',  #
        '23'  #
    ]
    forgetting_patients_ = [
        '1',  #
        # '2',  #
        # '3',  #
        # '5',  #
        # '6',  #
        # '8',  #
        # '9',  #
        # '10',  #
        # '13',  #
        # '14',  #
        # '16',  #
        # '17',  #
        # '18',  #
        # '19',  #
        # '20',  #
        # '21',  #
        # '22',  #
        # '23'  #
    ]
    remaining_patients_.remove(forgetting_patients_[0])
    samples, labels, n_samples, n_features = my_get_data(remaining_patients_, forgetting_patients_)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(samples)
    fig = plot_embedding(result, labels,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)


if __name__ == '__main__':
    main()
