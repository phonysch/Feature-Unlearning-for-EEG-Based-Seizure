import numpy as np
import torch.nn as nn
from PCT_net import PCT
import torch
import os
import csv
from copy import deepcopy
import json
import torch.utils.data as data
from load_data import PrepData


def get_raw_data(pat, settings):
    temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
    temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

    yield temp_pre_s, temp_inter_s


def load_data_kaggle(domain_labels, aug_pertur, *patients):
    """

    :param domain_labels: bool,
                          if 'True' attaching with domain labels
                          ('0' for remaining patients, '1' for excluded patients), ohterwise not.
    :param aug_pertur: bool,
                       if 'Ture' using part of excluded patient's data and
                       augmenting unlearning data in unlearingin phase, otherwise using all excluded patient's data and
                       do not augment unlearning data.
    :param patients: str in list
                     1) len(patients) == 1, original patients (all patients)
                     2) len(patinets) == 2, [0]: remaining patients; [1]: excluded patients
    :return: if len(patients) == 1, return original patients' dataloder in list;
             eilf len(patients) == 1, return remaining patients' dataloader in list
                                      and excluded patient's dataloader in list
    """
    dataset = 'Kaggle2014Pred'
    with open('SETTINGS_%s.json' % dataset) as f:
        settings = json.load(f)

    if len(patients) == 1:  # for training source model
        dataloader_list = []
        for ind, pat in enumerate(patients[0]):
            temp_pre_s, temp_inter_s = get_raw_data(pat, settings)

            samples_0 = np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32)
            samples_1 = temp_inter_s[:, np.newaxis, :, :].astype(np.float32)

            # # balance the two classes
            if len(samples_1) >= len(samples_0):
                np.random.shuffle(samples_1)
                samples_1 = samples_1[:len(samples_0)]
            else:
                np.random.shuffle(samples_0)
                samples_0 = samples_0[:len(samples_1)]

            labels_0 = np.ones(len(samples_0)).astype(np.int64)
            labels_1 = np.zeros(len(samples_1)).astype(np.int64)

            samples_0 = torch.from_numpy(samples_0)
            samples_1 = torch.from_numpy(samples_1)
            labels_0 = torch.from_numpy(labels_0)
            labels_1 = torch.from_numpy(labels_1)

            train_samples = torch.cat((samples_0, samples_1))
            train_labels = torch.cat((labels_0, labels_1))

            train_set = data.TensorDataset(train_samples, train_labels)
            train_loader = data.DataLoader(
                dataset=train_set,
                shuffle=True,
                batch_size=32,
                num_workers=0
            )

            dataloader_list.append(train_loader)

        return dataloader_list

    elif len(patients) == 2:  # for training unlearning model
        # for remaining patients
        dataloader_list_remaining, dataloader_list_excluded = [], []
        for ind, pat in enumerate(patients[0]):
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            samples_0 = np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32)
            samples_1 = np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32)

            # # balance the two classes
            if len(samples_1) >= len(samples_0):
                np.random.shuffle(samples_1)
                samples_1 = samples_1[:len(samples_0)]
            else:
                np.random.shuffle(samples_0)
                samples_0 = samples_0[:len(samples_1)]

            labels_0 = np.ones(len(samples_0)).astype(np.int64)
            labels_1 = np.zeros(len(samples_1)).astype(np.int64)

            samples_0 = torch.from_numpy(samples_0)
            samples_1 = torch.from_numpy(samples_1)
            labels_0 = torch.from_numpy(labels_0)
            labels_1 = torch.from_numpy(labels_1)

            train_samples = torch.cat((samples_0, samples_1))
            train_labels = torch.tensor((labels_0, labels_1))

            train_set = data.TensorDataset(train_samples, train_labels)
            train_loader = data.DataLoader(
                dataset=train_set,
                shuffle=True,
                batch_size=32,
                num_workers=0
            )

            dataloader_list_remaining.append(train_loader)

        # for excluded patient
        for ind, pat in enumerate(patients[1]):
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            samples_0 = np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32)
            samples_1 = np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32)

            # # balance the two classes
            if len(samples_1) >= len(samples_0):
                np.random.shuffle(samples_1)
                samples_1 = samples_1[:len(samples_0)]
            else:
                np.random.shuffle(samples_0)
                samples_0 = samples_0[:len(samples_1)]

            labels_0 = np.ones(len(samples_0)).astype(np.int64)
            labels_1 = np.zeros(len(samples_1)).astype(np.int64)

            samples_0 = torch.from_numpy(samples_0)
            samples_1 = torch.from_numpy(samples_1)
            labels_0 = torch.from_numpy(labels_0)
            labels_1 = torch.from_numpy(labels_1)

            train_samples = torch.cat((samples_0, samples_1))
            train_labels = torch.tensor((labels_0, labels_1))

            train_set = data.TensorDataset(train_samples, train_labels)
            train_loader = data.DataLoader(
                dataset=train_set,
                shuffle=True,
                batch_size=32,
                num_workers=0
            )

            dataloader_list_excluded.append(train_loader)
        return dataloader_list_remaining, dataloader_list_excluded
    else:
        print('unexpected argument patients')
        exit()


if __name__ == '__main__':

    # widen = PCT()
    # w = widen.featurer
    # b = widen.classifier

    # c = torch.nn.CrossEntropyLoss()
    #
    # unlearning_logits_dynamic = torch.zeros((1, 2), dtype=torch.float32, device='cuda')
    # a = torch.zeros((1,), dtype=torch.int64, device='cuda')
    # b = torch.zeros((1, 1), dtype=torch.int64, device='cuda')
    #
    # loss1 = c(unlearning_logits_dynamic, a)  # correct
    # # loss2 = c(unlearning_logits_dynamic, b)  # wrong
    #
    # unlearning_model = PCT()
    # unlearning_model.load_state_dict(torch.load(os.path.join('models/shadow states/original_model/CCT',
    #                                                          'original_model_para.pth')))
    # name = 'stem.layers.conv_1.weight'
    # d = unlearning_model.featurer.state_dict()[name]

    # with open('results/shadow states/results.csv') as csvfile:
    #     csv_reader = csv.reader(csvfile)
    #     for row in csv_reader:
    #         print(row[5])

    # p = {'1': 1, '2': 2, '3': 3}
    # a = sum(p.values())
    #
    # torch.manual_seed(3407)

    # original_model = PCT()
    # original_model.load_state_dict(torch.load(os.path.join('models/retrain featurer/original_model/CCT',
    #                                                        'original_model_para.pth')))
    #
    # dic1 = original_model.state_dict()
    #
    # for text in dic1:
    #     a = dic1[text]

    # logits = np.array([[0.75, 0.25]*16]).reshape(16, 2)
    # logits = torch.from_numpy(logits)
    #
    # rescal_logits = []
    # t = 2
    #
    # for index1, text1 in enumerate(logits):
    #     sum_exp = sum(torch.exp(text1 / t))
    #     a = torch.exp(text1 / t) * text1
    #     for text2 in text1:
    #         exp_logit = torch.exp(text2 / t)
    #         rescal_logits.append((exp_logit / sum_exp).view(1, ))
    # rescal_logits = torch.cat(rescal_logits).view(logits.shape[0], logits.shape[1])

    # a = np.array([i for i in range(5)])

    # all_patients = [
    #     'Dog_1',
    #     'Dog_2',
    #     'Dog_3',
    #     'Dog_4',
    #     'Dog_5',
    #     # 'Patient_1',
    #     # 'Patient_2',
    # ]
    #
    # train_loader_list = load_data_kaggle(False, False, all_patients)

    # batch_size_settings = {'32': {'Dog_1': 11, 'Dog_2': 11, 'Dog_3': 32, 'Dog_4': 18, 'Dog_5': 10},
    #                        '64': {'Dog_1': 22, 'Dog_2': 22, 'Dog_3': 64, 'Dog_4': 36, 'Dog_5': 20}}
    #
    # pat = 'Dog_1'
    #
    # a = batch_size_settings['64'][pat]

    for i in range(1, 3):
        print(i)

    print('test')



