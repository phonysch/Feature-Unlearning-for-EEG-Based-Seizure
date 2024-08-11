import json
from load_data import PrepData
import numpy as np
from sklearn import manifold
import einops
import matplotlib.pyplot as plt
# from PCT_net import PCT
# from PCT_net_original import PCT
from PCT_net_test import PCT
import torch
import os
from CNN import CNN
import torch.utils.data as data
import pandas as pd
from STMLP import STMLP
from Transformer import ViT
from STMLP_configs import configs


def get_unlearning_remaining_data_dots(unlearning_patients, remaining_patients, number):
    data_dots = [[], []]
    with open('SETTINGS_%s.json' % 'CHBMIT') as f:
        settings = json.load(f)

    # unlearning dots from patient 2 seizure data
    append_index = 0
    for index, pat in enumerate(unlearning_patients):  # '10' -> 1, 0
        temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
        temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

        data_dots[0].append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
        data_dots[1].append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

        np.random.shuffle(data_dots[0][append_index])
        np.random.shuffle(data_dots[1][append_index])

        data_dots[0][append_index] = data_dots[0][append_index][0:number[0]]
        data_dots[1][append_index] = data_dots[1][append_index][0:number[0]]

        append_index += 1

    # remaining dots from other patients

    for index, pat in enumerate(remaining_patients):  # '10' -> 1, 0

        temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
        temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

        data_dots[0].append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
        data_dots[1].append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

        np.random.shuffle(data_dots[0][append_index])
        np.random.shuffle(data_dots[1][append_index])

        data_dots[0][append_index] = data_dots[0][append_index][0:number[1]]
        data_dots[1][append_index] = data_dots[1][append_index][0:number[1]]

        append_index += 1

    data_dots[0] = np.concatenate(data_dots[0])
    data_dots[1] = np.concatenate(data_dots[1])

    return data_dots


def get_remaining_data_dots_and_unlearning_features(remaining_patients, number):
    remaining_data_dots = [[], []]
    with open('SETTINGS_%s.json' % 'CHBMIT') as f:
        settings = json.load(f)

    # get remaining dots for inputting into the three models
    for index, pat in enumerate(remaining_patients):  # '10' -> 1, 0

        temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
        temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

        remaining_data_dots[0].append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
        remaining_data_dots[1].append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

        np.random.shuffle(remaining_data_dots[0][index])
        np.random.shuffle(remaining_data_dots[1][index])

        remaining_data_dots[0][index] = remaining_data_dots[0][index][0:number[1]]
        remaining_data_dots[1][index] = remaining_data_dots[1][index][0:number[1]]

    remaining_data_dots[0] = np.concatenate(remaining_data_dots[0])
    remaining_data_dots[1] = np.concatenate(remaining_data_dots[1])

    # get unlearning features from the outputs of the three model
    forgetting_features = []

    # data_path = 'results/feature records'
    data_path = 'results/feature records'
    # # each fold
    for i in range(5):

        # # each model
        for t in ['original', 'unlearned', 'retrained']:
            temp = pd.read_csv(os.path.join(data_path, f'{t} model/feature distribution_{i - 5}.csv'))
            temp = temp.drop('Unnamed: 0', axis=1)

            len_columns = temp.shape[1]
            label_index = str(len_columns - 1)
            # the last column is labels
            drop_list_p = [i for i, t in enumerate(np.array((temp[label_index] == 0))) if t]
            drop_list_i = [i for i, t in enumerate(np.array((temp[label_index] == 1))) if t]
            pre = temp.drop(drop_list_p, axis=0)
            inter = temp.drop(drop_list_i, axis=0)
            pre = pre.drop(label_index, axis=1)
            inter = inter.drop(label_index, axis=1)
            pre = pre.to_numpy()
            inter = inter.to_numpy()
            np.random.shuffle(pre)
            np.random.shuffle(inter)

            forgetting_features.append(pre[0:number[0]])
            forgetting_features.append(inter[0:number[0]])

    return remaining_data_dots, forgetting_features


def my_t_SNE(data_dots, number):

    len_pre = len(data_dots[0])
    data_dots = np.concatenate(data_dots)
    data_dots = einops.rearrange(data_dots, 'n i l w -> n (i w l)')

    tsne = manifold.TSNE(n_components=2, perplexity=20, n_iter=1000, learning_rate=10)
    data_tsne = tsne.fit_transform(data_dots)

    data_min, data_max = data_tsne.min(0), data_tsne.max(0)
    data_norm = (data_tsne - data_min) / (data_max - data_min)

    unlearning_pre_dots = data_norm[0:number[0]]
    unlearning_inter_dots = data_norm[len_pre:len_pre+number[0]]

    remaining_pre_dots = data_norm[number[0]:len_pre]
    remaining_inter_dots = data_norm[len_pre+number[0]:]

    unlearning_pre_dots_x, unlearning_pre_dots_y = unlearning_pre_dots[:, 0], unlearning_pre_dots[:, 1]
    unlearning_inter_dots_x, unlearning_inter_dots_y = unlearning_inter_dots[:, 0], unlearning_inter_dots[:, 1]

    remaining_pre_dots_x, remaining_pre_dots_y = remaining_pre_dots[:, 0], remaining_pre_dots[:, 1]
    remaining_inter_dots_x, remaining_inter_dots_y = remaining_inter_dots[:, 0], remaining_inter_dots[:, 1]

    plt.scatter(unlearning_pre_dots_x, unlearning_pre_dots_y, marker='*', color='r')
    plt.scatter(unlearning_inter_dots_x, unlearning_inter_dots_y, marker='*', color='g')
    plt.scatter(remaining_pre_dots_x, remaining_pre_dots_y, marker='s', color='r')
    plt.scatter(remaining_inter_dots_x, remaining_inter_dots_y, marker='s', color='g')

    plt.show()

    print('my_t_SNE()')


def feature_projection_distribution(data_dots, number):

    models = {'MLP': STMLP(**configs["ST-MLP"]), 'ViT': ViT(), 'PCT': PCT(), 'CNN': CNN()}
    model_name = 'PCT'
    retrained_model = models[model_name]
    para_path = f'models/retrain/{model_name}'
    retrained_model.load_state_dict(torch.load(os.path.join(para_path, 'except_2_para.pth')))

    len_pre = len(data_dots[0])
    data_dots = np.concatenate(data_dots).astype(np.float32)
    labels = np.ones(len(data_dots))
    data_dots = torch.from_numpy(data_dots)
    labels = torch.from_numpy(labels)
    data_dots_set = data.TensorDataset(data_dots, labels)
    data_dots_loader = data.DataLoader(
        dataset=data_dots_set,
        shuffle=False,
        batch_size=64,
        num_workers=0
    )

    all_features = []
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    retrained_model.to(device)
    retrained_model.eval()
    with torch.no_grad():
        for dots, labels in data_dots_loader:
            dots = dots.to(device)

            features = retrained_model.featurer(dots)
            # features = retrained_model(dots)
            all_features.append(features)

        all_features = torch.cat(all_features)
        all_features = all_features.cpu().numpy()

    # if model_name == 'PCT':
    #     data_min, data_max = all_features.min(0), all_features.max(0)
    #     data_norm = (all_features - data_min) / (data_max - data_min)
    #
    # elif model_name == 'CNN':
    #     # data_dots = np.concatenate(data_dots)
    #     data_dots = einops.rearrange(data_dots, 'n i l w -> n (i w l)')
    #
    #     tsne = manifold.TSNE(n_components=2, perplexity=20, n_iter=1000, learning_rate=10)
    #     data_tsne = tsne.fit_transform(data_dots)
    #
    #     data_min, data_max = data_tsne.min(0), data_tsne.max(0)
    #     data_norm = (data_tsne - data_min) / (data_max - data_min)

    # data_dots = np.concatenate(data_dots)

    tsne = manifold.TSNE(n_components=2, perplexity=20, n_iter=1000, learning_rate=10)
    feature_tsne = tsne.fit_transform(all_features)

    feature_min, feature_max = feature_tsne.min(0), feature_tsne.max(0)
    feature_norm = (feature_tsne - feature_min) / (feature_max - feature_min)

    # data_norm = all_features

    unlearning_pre_dots = feature_norm[0:number[0]]
    unlearning_inter_dots = feature_norm[len_pre:len_pre + number[0]]

    remaining_pre_dots = feature_norm[number[0]:len_pre]
    remaining_inter_dots = feature_norm[len_pre + number[0]:]

    unlearning_pre_dots_x, unlearning_pre_dots_y = unlearning_pre_dots[:, 0], unlearning_pre_dots[:, 1]
    unlearning_inter_dots_x, unlearning_inter_dots_y = unlearning_inter_dots[:, 0], unlearning_inter_dots[:, 1]

    remaining_pre_dots_x, remaining_pre_dots_y = remaining_pre_dots[:, 0], remaining_pre_dots[:, 1]
    remaining_inter_dots_x, remaining_inter_dots_y = remaining_inter_dots[:, 0], remaining_inter_dots[:, 1]

    plt.scatter(unlearning_pre_dots_x, unlearning_pre_dots_y + 0.1, marker='*', color='r')
    plt.scatter(unlearning_inter_dots_x, unlearning_inter_dots_y + 0.1, marker='*', color='g')
    plt.scatter(remaining_pre_dots_x, remaining_pre_dots_y, marker='.', color='r')
    plt.scatter(remaining_inter_dots_x, remaining_inter_dots_y, marker='.', color='g')

    plt.show()

    len_remaining = len(remaining_pre_dots_x)
    len_unlearning = len(unlearning_pre_dots_x)

    if len_remaining > len_unlearning:
        frame = np.zeros(len_remaining - len_unlearning)
        unlearning_pre_dots_x = np.concatenate((unlearning_pre_dots_x, frame))
        unlearning_pre_dots_y = np.concatenate((unlearning_pre_dots_y, frame))
        unlearning_inter_dots_x = np.concatenate((unlearning_inter_dots_x, frame))
        unlearning_inter_dots_y = np.concatenate((unlearning_inter_dots_y, frame))
    elif len_remaining < len_unlearning:
        frame = np.zeros(len_unlearning - len_remaining)
        remaining_pre_dots_x = np.concatenate((remaining_pre_dots_x, frame))
        remaining_pre_dots_y = np.concatenate((remaining_pre_dots_y, frame))
        remaining_inter_dots_x = np.concatenate((remaining_inter_dots_x, frame))
        remaining_inter_dots_y = np.concatenate((remaining_inter_dots_y, frame))
    else:
        pass

    head = ['u_p_x', 'u_p_y', 'u_i_x', 'u_i_y', 'r_p_x', 'r_p_y', 'r_i_x', 'r_i_y']
    data_save = np.concatenate((unlearning_pre_dots_x.reshape(-1, 1), unlearning_pre_dots_y.reshape(-1, 1),
                                unlearning_inter_dots_x.reshape(-1, 1), unlearning_inter_dots_y.reshape(-1, 1),
                                remaining_pre_dots_x.reshape(-1, 1), remaining_pre_dots_y.reshape(-1, 1),
                                remaining_inter_dots_x.reshape(-1, 1), remaining_inter_dots_y.reshape(-1, 1)), axis=1)
    results = pd.DataFrame(data_save, columns=head)
    results.to_csv('data_dots.csv')

    print('feature_projection_distribution()')


def feature_projection_distribution_new(remaining_data_dots, forgetting_features, number):

    # remaining data dots to features
    remaining_features = []
    for i, t in enumerate(['original', 'unlearned', 'retrain']):
        model = PCT()
        if t not in ['retrain']:
            model.load_state_dict(torch.load(os.path.join(f'models/retrain featurer/{t}_model/CHBMIT/PCT',
                                                          f'test_th_{-5}_{t}_model_para.pth')))
        else:
            model.load_state_dict(torch.load('models/retrain/PCT/except_1_para.pth'))

        data_dots = np.concatenate(remaining_data_dots).astype(np.float32)
        labels = np.ones(len(data_dots))
        data_dots = torch.from_numpy(data_dots)
        labels = torch.from_numpy(labels)
        data_dots_set = data.TensorDataset(data_dots, labels)
        data_dots_loader = data.DataLoader(
            dataset=data_dots_set,
            shuffle=False,
            batch_size=64,
            num_workers=0
        )

        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        with torch.no_grad():
            for dots, labels in data_dots_loader:
                dots = dots.to(device)

                features = model.featurer(dots)
                remaining_features.append(features)

    remaining_features = torch.cat(remaining_features)
    remaining_features = remaining_features.cpu().numpy()

    # t-SNE
    features = np.concatenate((remaining_features, np.concatenate(forgetting_features)))

    # # feature not in N*2
    tsne = manifold.TSNE(n_components=2, perplexity=20, n_iter=1000, learning_rate=10)
    features_tsne = tsne.fit_transform(features)

    data_min, data_max = features_tsne.min(0), features_tsne.max(0)
    features_norm = (features_tsne - data_min) / (data_max - data_min)

    # # # feature in N*2
    # data_min, data_max = features.min(0), features.max(0)
    # features_norm = (features - data_min) / (data_max - data_min)

    # plot each folds data features of unlearning patient (patient 1), and the remaining data features
    fig, ax = plt.subplots(1, 3)

    for i in range(6):
        # # plot the first figure (original model feature distribution)
        if i == 0:
            for j in range(int(len(remaining_data_dots)/2)):  # number of remaining patients
                # # plot the first figure (original model feature distribution)
                # # # original remaining pre
                ax[0].scatter(features_norm[(j * 6 + 0) * number[1]:(j * 6 + 1) * number[1]][:, 0],
                              features_norm[(j * 6 + 0) * number[1]:(j * 6 + 1) * number[1]][:, 1],
                              marker='s', color='r')
                # # # original remaining inter
                ax[0].scatter(features_norm[(j * 6 + 1) * number[1]:(j * 6 + 2) * number[1]][:, 0],
                              features_norm[(j * 6 + 1) * number[1]:(j * 6 + 2) * number[1]][:, 1],
                              marker='s', color='g')

                # # plot the second figure (unlearned model feature distribution)
                # # # unlearned remaining pre and inter
                ax[1].scatter(features_norm[(j * 6 + 2) * number[1]:(j * 6 + 3) * number[1]][:, 0],
                              features_norm[(j * 6 + 2) * number[1]:(j * 6 + 3) * number[1]][:, 1],
                              marker='s', color='r')
                # # # original remaining inter
                ax[1].scatter(features_norm[(j * 6 + 3) * number[1]:(j * 6 + 4) * number[1]][:, 0],
                              features_norm[(j * 6 + 3) * number[1]:(j * 6 + 4) * number[1]][:, 1],
                              marker='s', color='g')

                # # plot the third figure (retrained model feature distribution)
                # # # retrained remaining pre and inter
                ax[2].scatter(features_norm[(j * 6 + 4) * number[1]:(j * 6 + 5) * number[1]][:, 0],
                              features_norm[(j * 6 + 4) * number[1]:(j * 6 + 5) * number[1]][:, 1],
                              marker='s', color='r')
                # # # original remaining inter
                ax[2].scatter(features_norm[(j * 6 + 5) * number[1]:(j * 6 + 6) * number[1]][:, 0],
                              features_norm[(j * 6 + 5) * number[1]:(j * 6 + 6) * number[1]][:, 1],
                              marker='s', color='g')
        else:
            if i == 1:
                base = int(len(remaining_data_dots)/2) * number[1] * 2 * 3
                shift = number[0] * 2 * 3 * (i - 1)
                # # # original unlearning pre and inter (each folds, add 0.05 to the ordinate for clear visualization)
                ax[0].scatter(features_norm[base + shift + 0 * 20:base + shift + 1 * 20][:, 0],
                              features_norm[base + shift + 0 * 20:base + shift + 1 * 20][:, 1]
                              # + 0.05 * i
                              , marker='*', color='r')
                ax[0].scatter(features_norm[base + shift + 1 * 20:base + shift + 2 * 20][:, 0],
                              features_norm[base + shift + 1 * 20:base + shift + 2 * 20][:, 1]
                              # + 0.05 * i
                              , marker='*', color='g')

                # # plot the second figure (unlearned model feature distribution)
                # # # unlearned unlearning pre and inter
                ax[1].scatter(features_norm[base + shift + 2 * 20:base + shift + 3 * 20][:, 0],
                              features_norm[base + shift + 2 * 20:base + shift + 3 * 20][:, 1]
                              # + 0.05 * i
                              , marker='*', color='r')
                ax[1].scatter(features_norm[base + shift + 3 * 20:base + shift + 4 * 20][:, 0],
                              features_norm[base + shift + 3 * 20:base + shift + 4 * 20][:, 1]
                              # + 0.05 * i
                              , marker='*', color='g')

                # # plot the third figure (retrained model feature distribution)
                # # # retrained unlearning pre and inter
                ax[2].scatter(features_norm[base + shift + 4 * 20:base + shift + 5 * 20][:, 0],
                              features_norm[base + shift + 4 * 20:base + shift + 5 * 20][:, 1]
                              # + 0.05 * i
                              , marker='*', color='r')
                ax[2].scatter(features_norm[base + shift + 5 * 20:base + shift + 6 * 20][:, 0],
                              features_norm[base + shift + 5 * 20:base + shift + 6 * 20][:, 1]
                              # + 0.05 * i
                              , marker='*', color='g')

    plt.show()

    # save the features after performing t-SNE
    rows, columns = [], []
    for i in range(len(features_norm)):
        if i <= 15:
            columns.append(str(i+1))
        rows.append(str(i+1))

    features_norm = pd.DataFrame(features_norm, index=rows, columns=columns)
    features_norm.to_csv('results/feature records/feature_tsne_norm.csv')

    print('feature_projection_distribution_new()')


if __name__ == '__main__':

    remaining_patients_ = [
        '1',  #
        '2',
        '3',  #
        '5',  #
        '6',  #
        '8',  #
        '9',
        '10',  #
        '13',  #
        '14',  #
        '16',  #
        '17',
        '18',  #
        '19',
        '20',  #
        '21',
        '23'  #
    ]
    unlearning_patients_ = ['2']
    remaining_patients_.remove(unlearning_patients_[0])

    # plot Fig. 1
    each_class_dots_num_unlearining = 75
    each_class_dots_num_remaining = 10
    each_class_dots_num = [each_class_dots_num_unlearining, each_class_dots_num_remaining]

    # load data dots of unlearning and remaining samples
    data_dots_ = get_unlearning_remaining_data_dots(unlearning_patients_, remaining_patients_, each_class_dots_num)

    # # process data dots by t-SNE dimension reduction, and plot processed data dots
    # my_t_SNE(data_dots_, each_class_dots_num)

    # plot features
    feature_projection_distribution(data_dots_, each_class_dots_num)

    # # plot original, unlearned, and retrained feature distribution
    # each_class_dots_num_unlearining = 75
    # each_class_dots_num_remaining = 20
    # each_class_dots_num = [each_class_dots_num_unlearining, each_class_dots_num_remaining]
    # remaining_data_dots_, forgetting_features_ = \
    #     get_remaining_data_dots_and_unlearning_features(remaining_patients_, each_class_dots_num)
    #
    # feature_projection_distribution_new(remaining_data_dots_, forgetting_features_, each_class_dots_num)

    print('plot dots unlearning and remaining.py')
