import os.path
import csv
from sklearn import metrics
import torch.nn.functional as func
import json
from load_data import PrepData
import numpy as np
import torch
import torch.utils.data as data
from copy import deepcopy
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
import torch.optim as optim


def load_data(domain_labels=False, *patients):
    """
    get patients' data and obtain a dataloader by using the data
    :param domain_labels: if True appending domain labels, otherwise no domain labels
    :param patients: must str in list.
                     1) one variable: original patients
                     2) two variables: remaining patients and unleanring patient
    :return: 1) one variable: return a dataloader used for training original model
             2) two variables: return a dataloader used for training unlearning model
    """
    dataset = 'CHBMIT'
    with open('SETTINGS_%s.json' % dataset) as f:
        settings = json.load(f)

    # data -> set -> loader (for training an original model)
    if len(patients) == 1:
        patient_number = {}
        samples_0, labels_0, samples_1, labels_1 = [], [], [], []
        for index, pat in enumerate(patients[0]):  # '10' -> 1, 0
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            if dataset == 'CHBMIT':
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

                # # balance the two classes
                if len(samples_1[index]) >= len(samples_0[index]):
                    np.random.shuffle(samples_1[index])
                    samples_1[index] = samples_1[index][:len(samples_0[index])]
                else:
                    np.random.shuffle(samples_0[index])
                    samples_0[index] = samples_0[index][:len(samples_1[index])]

            elif dataset == 'Kaggle2014Pred':
                # not shffule reslut from using all data (pre and inter approximately equal)
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))

            else:
                raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                                 "but got dataset={}".format(dataset))

            patient_number[pat] = len(samples_0[index]) + len(samples_1[index])
            labels_0.append(np.ones(len(samples_0[index])).astype(np.int64))
            labels_1.append(np.zeros(len(samples_1[index])).astype(np.int64))

        # if not os.path.exists(f'number_samples_patient_{dataset}.csv'):
        #     with open(f'number_samples_patient_{dataset}.csv', 'a+', encoding='utf8', newline='') as file:
        #         writer = csv.writer(file)
        #         for key, value in patient_number.items():
        #             content = [key, value]
        #             writer.writerow(content)

        train_samples = np.concatenate((np.concatenate(samples_0), np.concatenate(samples_1)))
        train_labels = np.concatenate((np.concatenate(labels_0), np.concatenate(labels_1)))

        train_samples = torch.from_numpy(train_samples)
        train_labels = torch.from_numpy(train_labels)

        train_set = data.TensorDataset(train_samples, train_labels)
        train_loader = data.DataLoader(
            dataset=train_set,
            shuffle=True,
            batch_size=64,
            num_workers=0
        )

        return train_loader

    elif len(patients) == 2:
        patient_number = {}
        with open(f'number_samples_patient_{dataset}.csv') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                patient_number[row[0]] = int(row[1])

        remaining_patients = patients[0]
        unlearning_patient = patients[1]

        unlearning_data_rate = 0.5
        # balance the remaining data and unlearning data
        unlearning_number = 0.0
        for patient in unlearning_patient:
            unlearning_number += patient_number[patient]
        remaining_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
                                                                     - unlearning_number)
        len_remaining = 0
        len_unlearning = 0

        samples_0, labels_0, samples_1, labels_1 = [], [], [], []
        append_index = 0
        # get remaining patients' data
        print("get subset of remaining patients' data")
        for index, pat in enumerate(remaining_patients):  # '10' -> 1, 0
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            if dataset == 'CHBMIT':
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

                # # balance the two classes
                if len(samples_1[index]) >= len(samples_0[index]):
                    np.random.shuffle(samples_1[index])
                    samples_1[index] = samples_1[index][:len(samples_0[index])]
                else:
                    np.random.shuffle(samples_0[index])
                    samples_0[index] = samples_0[index][:len(samples_1[index])]

            elif dataset == 'Kaggle2014Pred':
                # not shffule reslut from using all data (pre and inter approximately equal)
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))

                np.random.shuffle(samples_0)
                np.random.shuffle(samples_1)

            else:
                raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                                 "but got dataset={}".format(dataset))

            # # get subset with remaining_rate of each remaining patient
            samples_0[index] = samples_0[index][:int(len(samples_0[index])*remaining_rate)]
            samples_1[index] = samples_1[index][:int(len(samples_1[index])*remaining_rate)]
            # # '0' and '1' for preictal and interictal segments of remaining patients, respectively
            labels_0.append(np.ones(len(samples_0[index])).astype(np.int64))
            labels_1.append(np.zeros(len(samples_1[index])).astype(np.int64))
            len_remaining += (len(samples_0[index]) + len(samples_1[index]))

            append_index += 1

        # get unlearning patients' data
        print("\nget unlearning patients' data")
        for index, pat in enumerate(unlearning_patient):  # '10' -> 1, 0
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            if dataset == 'CHBMIT':
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

                # # balance the two classes
                if len(samples_1[append_index]) >= len(samples_0[append_index]):
                    np.random.shuffle(samples_1[append_index])
                    samples_1[append_index] = samples_1[append_index][:len(samples_0[append_index])]
                else:
                    np.random.shuffle(samples_0[append_index])
                    samples_0[append_index] = samples_0[append_index][:len(samples_1[append_index])]

            elif dataset == 'Kaggle2014Pred':
                # not shffule reslut from using all data (pre and inter approximately equal)
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))

                np.random.shuffle(samples_0)
                np.random.shuffle(samples_1)

            else:
                raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                                 "but got dataset={}".format(dataset))

            # # # '2' and '3' for preictal and interictal segments of unlearning patient
            # labels_0[append_index] = np.ones(len(samples_0[append_index]), dtype=int) * 2
            # labels_1[append_index] = np.ones(len(samples_1[append_index]), dtype=int) * 3
            # # reverse labels
            labels_0.append(np.zeros(len(samples_0[append_index])).astype(np.int64))
            labels_1.append(np.ones(len(samples_1[append_index])).astype(np.int64))
            len_unlearning += (len(samples_0[append_index]) + len(samples_1[append_index]))

            append_index += 1

        unlearning_samples = np.concatenate((np.concatenate(samples_0), np.concatenate(samples_1)))
        unlearning_labels = np.concatenate((np.concatenate(labels_0), np.concatenate(labels_1)))

        unlearning_samples = torch.from_numpy(unlearning_samples)
        unlearning_labels = torch.from_numpy(unlearning_labels)

        if domain_labels:
            # # '0' for samples from remaining patients, '1' for samples from unlearning patient.
            unlearning_labels_domain = np.concatenate((np.zeros(len_remaining, dtype=np.int64),
                                                       np.ones(len_unlearning, dtype=np.int64)))

            unlearning_labels_domain = torch.from_numpy(unlearning_labels_domain)

            unlearning_set = data.TensorDataset(unlearning_samples, unlearning_labels, unlearning_labels_domain)
            unlearning_loader = data.DataLoader(
                dataset=unlearning_set,
                shuffle=True,
                batch_size=64,
                num_workers=0
            )

            return unlearning_loader

        else:
            unlearning_set = data.TensorDataset(unlearning_samples, unlearning_labels)
            unlearning_loader = data.DataLoader(
                dataset=unlearning_set,
                shuffle=True,
                batch_size=64,
                num_workers=0
            )

            return unlearning_loader


def load_data_balance(domain_labels, aug_pertur, dataset, *patients):
    """
    get patients' data and obtain a dataloader by using the data
    :param domain_labels: if True appending domain labels, otherwise no domain labels
    :param patients: must str in list.
                    1) one variable: original patients
                    2) two variables: remaining patients and unleanring patient
    :param aug_pertur: int; 0: augment origianl data, 1: perturb original data (mask some channels),
                            2: original data without any method
    :param dataset:
    :return: 1) one variable: return a dataloader used for training original model
             2) two variables: return a dataloader used for training unlearning model
    """
    # with open('SETTINGS_%s.json' % dataset) as f:
    #     settings = json.load(f)
    #
    # data -> set -> loader (for training a original model)
    # if len(patients) == 1:
    #     patient_number = {}
    #     samples_0, labels_0, samples_1, labels_1 = [], [], [], []
    #     for index, pat in enumerate(patients[0]):  # '10' -> 1, 0
    #         temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
    #         temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()
    #
    #         if dataset == 'CHBMIT':
    #             samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
    #             samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))
    #
    #         elif dataset == 'Kaggle2014Pred':
    #             # not shffule reslut from using all data (pre and inter approximately equal)
    #             samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
    #             samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))
    #
    #         else:
    #             raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
    #                              "but got dataset={}".format(dataset))
    #
    #         # balance the two classes (useless for kaggle dataset)
    #         if len(samples_1[index]) >= len(samples_0[index]):
    #             np.random.shuffle(samples_1[index])
    #             samples_1[index] = samples_1[index][:len(samples_0[index])]
    #         else:
    #             np.random.shuffle(samples_0[index])
    #             samples_0[index] = samples_0[index][:len(samples_1[index])]
    #
    #         patient_number[pat] = len(samples_0[index]) + len(samples_1[index])
    #
    #         labels_0.append(np.ones(len(samples_0[index])).astype(np.int64))
    #         labels_1.append(np.zeros(len(samples_1[index])).astype(np.int64))
    #
    #     if not os.path.exists(f'number_samples_patient_{dataset}.csv'):
    #         with open(f'number_samples_patient_{dataset}.csv', 'a+', encoding='utf8', newline='') as file:
    #             writer = csv.writer(file)
    #             for key, value in patient_number.items():
    #                 content = [key, value]
    #                 writer.writerow(content)
    #
    #     train_samples, train_labels = [], []
    #     for j in range(len(samples_0)):
    #         # # append preictal
    #         train_samples.append(samples_0[j])
    #         train_labels.append(labels_0[j])
    #
    #         # # append interictal
    #         train_samples.append(samples_1[j])
    #         train_labels.append(labels_1[j])
    #
    #     train_samples = torch.from_numpy(np.concatenate(train_samples))
    #     train_labels = torch.from_numpy(np.concatenate(train_labels))
    #
    #     train_set = data.TensorDataset(train_samples, train_labels)
    #     train_loader = data.DataLoader(
    #         dataset=train_set,
    #         shuffle=True,
    #         batch_size=64,
    #         num_workers=0
    #     )
    #
    #     return train_loader
    #
    # elif len(patients) == 2:
    #     patient_number = {}
    #     with open(f'number_samples_patient_{dataset}.csv') as csvfile:
    #         csv_reader = csv.reader(csvfile)
    #         for row in csv_reader:
    #             patient_number[row[0]] = int(row[1])
    #
    #     # get remaining patients' data
    #     remaining_patients = patients[0]
    #     unlearning_patient = patients[1]
    #     if aug_pertur == 0:
    #         # break the balance rule between the remaining data and unlearning data
    #         unlearning_data_rate = 0.5
    #         # balance the remaining data and unlearning data
    #         unlearning_number = 0.0
    #         for patient in unlearning_patient:
    #             unlearning_number += patient_number[patient]
    #         remaining_data_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
    #                                                                           - unlearning_number)
    #     elif aug_pertur == 1:
    #         unlearning_data_rate = 1.0
    #         unlearning_number = 0.0
    #         for patient in unlearning_patient:
    #             unlearning_number += patient_number[patient]
    #         remaining_data_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
    #                                                                           - unlearning_number)
    #     elif aug_pertur == 2:
    #         unlearning_data_rate = 0.1
    #         unlearning_number = 0.0
    #         for patient in unlearning_patient:
    #             unlearning_number += patient_number[patient]
    #         remaining_data_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
    #                                                                           - unlearning_number)
    #
    #     else:
    #         raise ValueError("aug_pertur expected specified option in '0', '1' and '2', "
    #                          "but got aug_pertur={}".format(aug_pertur))
    #
    #     print("get subset of remaining patients' data")
    #     append_index, len_remaining, len_unlearning = 0, 0, 0
    #     samples_0, labels_0, samples_1, labels_1 = [], [], [], []
    #     for index, pat in enumerate(remaining_patients):  # '10' -> 1, 0
    #         temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
    #         temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()
    #
    #         if dataset == 'CHBMIT':
    #             samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
    #             samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))
    #
    #         elif dataset == 'Kaggle2014Pred':
    #             samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
    #             samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))
    #
    #             np.random.shuffle(samples_0[index])
    #             np.random.shuffle(samples_1[index])
    #
    #         else:
    #             raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
    #                              "but got dataset={}".format(dataset))
    #
    #         # # balance the two classes
    #         if len(samples_1[index]) >= len(samples_0[index]):
    #             np.random.shuffle(samples_1[index])
    #             samples_1[index] = samples_1[index][:len(samples_0[index])]
    #         else:
    #             np.random.shuffle(samples_0[index])
    #             samples_0[index] = samples_0[index][:len(samples_1[index])]
    #
    #         samples_0[index] = samples_0[index][:int(len(samples_0[index]) * remaining_data_rate)]
    #         samples_1[index] = samples_1[index][:int(len(samples_1[index]) * remaining_data_rate)]
    #
    #         # '1' and '0' for preictal and interictal segments of remaining patients, respectively
    #         labels_0.append(np.ones(len(samples_0[index])).astype(np.int64))
    #         labels_1.append(np.zeros(len(samples_1[index])).astype(np.int64))
    #
    #         len_remaining += (len(samples_0[index]) + len(samples_1[index]))
    #
    #         append_index += 1
    #
    #     # get unlearning patients' data
    #     print("\nget unlearning patients' data")
    #     for index, pat in enumerate(unlearning_patient):  # '10' -> 1, 0
    #         temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
    #         temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()
    #
    #         if dataset == 'CHBMIT':
    #             samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
    #             samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))
    #
    #         elif dataset == 'Kaggle2014Pred':
    #             samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
    #             samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))
    #
    #             np.random.shuffle(samples_0[append_index])
    #             np.random.shuffle(samples_1[append_index])
    #
    #         else:
    #             raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
    #                              "but got dataset={}".format(dataset))
    #
    #         # # balance the two classes
    #         if len(samples_1[append_index]) >= len(samples_0[append_index]):
    #             np.random.shuffle(samples_1[append_index])
    #             samples_1[append_index] = samples_1[append_index][:len(samples_0[append_index])]
    #         else:
    #             np.random.shuffle(samples_0[append_index])
    #             samples_0[append_index] = samples_0[append_index][:len(samples_1[append_index])]
    #
    #         # used for 'using part of unlearning patient's data plan'
    #         samples_0[append_index] = samples_0[append_index][:int(len(samples_0[append_index]) * unlearning_data_rate)]
    #         samples_1[append_index] = samples_1[append_index][:int(len(samples_1[append_index]) * unlearning_data_rate)]
    #
    #         # # reverse labels
    #         labels_0.append(np.zeros(len(samples_0[append_index])).astype(np.int64))
    #         labels_1.append(np.ones(len(samples_1[append_index])).astype(np.int64))
    #
    #         len_unlearning += (len(samples_0[append_index]) + len(samples_1[append_index]))
    #
    #         append_index += 1
    #
    #     # key coding
    #     unlearning_samples, unlearning_labels = [], []
    #     for j in range(len(samples_0)):
    #         # # append preictal
    #         unlearning_samples.append(samples_0[j])
    #         unlearning_labels.append(labels_0[j])
    #
    #         # # append interictal
    #         unlearning_samples.append(samples_1[j])
    #         unlearning_labels.append(labels_1[j])
    #     # key coding
    #
    #     unlearning_samples = np.concatenate(unlearning_samples)
    #     unlearning_labels = np.concatenate(unlearning_labels)
    #
    #     unlearning_samples = torch.from_numpy(unlearning_samples)
    #     unlearning_labels = torch.from_numpy(unlearning_labels)
    #
    #     if domain_labels:
    #         # # '0' for samples from remaining patients, '1' for samples from unlearning patient.
    #         unlearning_labels_domain = np.concatenate((np.zeros(len_remaining, dtype=np.int64),
    #                                                    np.ones(len_unlearning, dtype=np.int64)))
    #
    #         unlearning_labels_domain = torch.from_numpy(unlearning_labels_domain)
    #
    #         unlearning_set = data.TensorDataset(unlearning_samples, unlearning_labels, unlearning_labels_domain)
    #         unlearning_loader = data.DataLoader(
    #             dataset=unlearning_set,
    #             shuffle=True,
    #             batch_size=64,
    #             num_workers=0
    #         )
    #
    #         return unlearning_loader
    #
    #     else:
    #         unlearning_set = data.TensorDataset(unlearning_samples, unlearning_labels)
    #         unlearning_loader = data.DataLoader(
    #             dataset=unlearning_set,
    #             shuffle=True,
    #             batch_size=64,
    #             num_workers=0
    #         )
    #
    #         return unlearning_loader

    with open('SETTINGS_%s.json' % dataset) as f:
        settings = json.load(f)

    # data -> set -> loader (for training an original model)
    if len(patients) == 1:
        patient_number = {}
        samples_0, labels_0, samples_1, labels_1 = [], [], [], []
        for index, pat in enumerate(patients[0]):  # '10' -> 1, 0
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            if dataset == 'CHBMIT':
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

            elif dataset == 'Kaggle2014Pred':
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))

            else:
                raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                                 "but got dataset={}".format(dataset))

            # # balance the two classes
            if len(samples_1[index]) >= len(samples_0[index]):
                np.random.shuffle(samples_1[index])
                samples_1[index] = samples_1[index][:len(samples_0[index])]
            else:
                np.random.shuffle(samples_0[index])
                samples_0[index] = samples_0[index][:len(samples_1[index])]

            patient_number[pat] = len(samples_0[index]) + len(samples_1[index])

            labels_0.append(np.ones(len(samples_0[index]), dtype=np.int64))
            labels_1.append(np.zeros(len(samples_1[index]), dtype=np.int64))

        if not os.path.exists(f'number_samples_patient_{dataset}.csv'):
            with open(f'number_samples_patient_{dataset}.csv', 'a+', encoding='utf8', newline='') as file:
                writer = csv.writer(file)
                for key, value in patient_number.items():
                    content = [key, value]
                    writer.writerow(content)

        train_samples = np.concatenate((np.concatenate(samples_0), np.concatenate(samples_1)))
        train_labels = np.concatenate((np.concatenate(labels_0), np.concatenate(labels_1)))

        train_samples = torch.from_numpy(train_samples)
        train_labels = torch.from_numpy(train_labels)

        train_set = data.TensorDataset(train_samples, train_labels)
        train_loader = data.DataLoader(
            dataset=train_set,
            shuffle=True,
            batch_size=64,
            num_workers=0
        )

        return train_loader

    elif len(patients) == 2:

        remaining_patients = patients[0]
        unlearning_patient = patients[1]
        # remaining_rate = 0.1
        len_remaining = 0
        len_unlearning = 0

        samples_0, labels_0, samples_1, labels_1 = [], [], [], []
        append_index = 0

        # get number of samples of each patient
        patient_number = {}
        with open(f'number_samples_patient_{dataset}.csv') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                patient_number[row[0]] = int(row[1])

        if aug_pertur == 0:
            # break the balance rule between the remaining data and unlearning data
            unlearning_data_rate = 0.1
            # remaining_rate = 0.1
            # balance the remaining data and unlearning data
            unlearning_number = 0.0
            for patient in unlearning_patient:
                unlearning_number += patient_number[patient]
            remaining_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
                                                                         - unlearning_number)
        elif aug_pertur == 1:
            unlearning_data_rate = 1.0
            unlearning_number = 0.0
            for patient in unlearning_patient:
                unlearning_number += patient_number[patient]
            remaining_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
                                                                         - unlearning_number)
        elif aug_pertur == 2:
            unlearning_data_rate = 0.5
            # balancing unlearning data and remaining data
            remaining_rate = 0.1
            # unlearning_number = 0.0
            # for patient in unlearning_patient:
            #     unlearning_number += patient_number[patient]
            # remaining_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
            #                                                              - unlearning_number)

        else:
            raise ValueError("aug_pertur expected specified option in '0', '1' and '2', "
                             "but got aug_pertur={}".format(aug_pertur))

        # get remaining patients' data
        print("get subset of remaining patients' data")
        for index, pat in enumerate(remaining_patients):  # '10' -> 1, 0
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            if dataset == 'CHBMIT':
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

                # # balance the two classes
                if len(samples_1[index]) >= len(samples_0[index]):
                    np.random.shuffle(samples_1[index])
                    samples_1[index] = samples_1[index][:len(samples_0[index])]
                else:
                    np.random.shuffle(samples_0[index])
                    samples_0[index] = samples_0[index][:len(samples_1[index])]

            elif dataset == 'Kaggle2014Pred':
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))

            else:
                raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                                 "but got dataset={}".format(dataset))

            # # get subset with remaining_rate of each remaining patient
            samples_0[index] = samples_0[index][:int(len(samples_0[index]) * remaining_rate)]
            samples_1[index] = samples_1[index][:int(len(samples_1[index]) * remaining_rate)]
            # # '0' and '1' for preictal and interictal segments of remaining patients, respectively
            labels_0.append(np.ones(len(samples_0[index]), dtype=np.int64))
            labels_1.append(np.zeros(len(samples_1[index]), dtype=np.int64))
            len_remaining += (len(samples_0[index]) + len(samples_1[index]))

            append_index += 1

        # get unlearning patients' data
        if domain_labels:  # for 'only use remaining samples to unlearning method'
            print("\nget unlearning patients' data")
            for index, pat in enumerate(unlearning_patient):  # '10' -> 1, 0
                temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
                temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

                if dataset == 'CHBMIT':
                    samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                    samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

                    # # balance the two classes
                    if len(samples_1[append_index]) >= len(samples_0[append_index]):
                        np.random.shuffle(samples_1[append_index])
                        samples_1[append_index] = samples_1[append_index][:len(samples_0[append_index])]
                    else:
                        np.random.shuffle(samples_0[append_index])
                        samples_0[append_index] = samples_0[append_index][:len(samples_1[append_index])]

                elif dataset == 'Kaggle2014Pred':
                    samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                    samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))

                else:
                    raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                                     "but got dataset={}".format(dataset))

                samples_0[append_index] = samples_0[append_index][:int(len(samples_0[append_index]) *
                                                                       unlearning_data_rate)]
                samples_1[append_index] = samples_1[append_index][:int(len(samples_1[append_index]) *
                                                                       unlearning_data_rate)]

                # # # '2' and '3' for preictal and interictal segments of unlearning patient
                # labels_0[append_index] = np.ones(len(samples_0[append_index]), dtype=int) * 2
                # labels_1[append_index] = np.ones(len(samples_1[append_index]), dtype=int) * 3
                # # # random labels
                # labels_0[append_index] = np.random.randint(0, 2, len(samples_0[append_index]), dtype=int)
                # labels_1[append_index] = np.random.randint(0, 2, len(samples_1[append_index]), dtype=int)

                # # reverse labels 1
                labels_0.append(np.zeros(len(samples_0[append_index]), dtype=np.int64))
                labels_1.append(np.ones(len(samples_1[append_index]), dtype=np.int64))

                # reverse labels 2
                len_pre = len(samples_0[append_index])
                len_inter = len(samples_1[append_index])
                labels_0.append(np.concatenate((np.zeros(int(len_pre/2), dtype=np.int64),
                                               np.ones(len_pre-int(len_pre/2), dtype=np.int64))))
                labels_1.append(np.concatenate((np.zeros(len_inter, dtype=np.int64),
                                                np.ones(len_inter-int(len_inter/2), dtype=np.int64))))

                len_unlearning += (len(samples_0[append_index]) + len(samples_1[append_index]))

                append_index += 1

        patients_data = {'p_s': samples_0, 'p_l': labels_0, 'i_s': samples_1, 'i_l': labels_1}

        unlearning_samples, unlearning_labels = [], []
        for j in range(len(patients_data['p_s'])):
            # # append preictal
            unlearning_samples.append(patients_data['p_s'][j])
            unlearning_labels.append(patients_data['p_l'][j])

            # # append interictal
            unlearning_samples.append(patients_data['i_s'][j])
            unlearning_labels.append(patients_data['i_l'][j])

        unlearning_samples = np.concatenate(unlearning_samples)
        unlearning_labels = np.concatenate(unlearning_labels)

        unlearning_samples = torch.from_numpy(unlearning_samples)
        unlearning_labels = torch.from_numpy(unlearning_labels)

        if domain_labels:
            # # '0' for samples from remaining patients, '1' for samples from unlearning patient.
            unlearning_labels_domain = np.concatenate((np.zeros(len_remaining, dtype=np.int64),
                                                       np.ones(len_unlearning, dtype=np.int64)))

            unlearning_labels_domain = torch.from_numpy(unlearning_labels_domain)

            unlearning_set = data.TensorDataset(unlearning_samples, unlearning_labels, unlearning_labels_domain)
            unlearning_loader = data.DataLoader(
                dataset=unlearning_set,
                shuffle=True,
                batch_size=64,
                num_workers=0
            )

            return unlearning_loader

        else:
            unlearning_set = data.TensorDataset(unlearning_samples, unlearning_labels)
            unlearning_loader = data.DataLoader(
                dataset=unlearning_set,
                shuffle=True,
                batch_size=64,
                num_workers=0
            )

            return unlearning_loader


def load_data_balance_not_mix_data(domain_labels, aug_pertur, dataset, test_th, *patients):
    """
    get patients' data and obtain a dataloader by using the data
    :param domain_labels: if True appending domain labels, otherwise no domain labels
    :param patients: must str in list.
                    1) one variable: original patients
                    2) two variables: remaining patients and unleanring patient
    :param aug_pertur: int; 0: augment origianl data, 1: perturb original data (mask some channels),
                            2: original data without any method
    :param dataset:
    :param test_th: the index of testing seizure data
    :return: 1) one variable: return a dataloader used for training original model
             2) two variables: return a dataloader used for training unlearning model
    """
    # with open('SETTINGS_%s.json' % dataset) as f:
    #     settings = json.load(f)
    #
    # data -> set -> loader (for training a original model)
    # if len(patients) == 1:
    #     patient_number = {}
    #     samples_0, labels_0, samples_1, labels_1 = [], [], [], []
    #     for index, pat in enumerate(patients[0]):  # '10' -> 1, 0
    #         temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
    #         temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()
    #
    #         if dataset == 'CHBMIT':
    #             samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
    #             samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))
    #
    #         elif dataset == 'Kaggle2014Pred':
    #             # not shffule reslut from using all data (pre and inter approximately equal)
    #             samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
    #             samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))
    #
    #         else:
    #             raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
    #                              "but got dataset={}".format(dataset))
    #
    #         # balance the two classes (useless for kaggle dataset)
    #         if len(samples_1[index]) >= len(samples_0[index]):
    #             np.random.shuffle(samples_1[index])
    #             samples_1[index] = samples_1[index][:len(samples_0[index])]
    #         else:
    #             np.random.shuffle(samples_0[index])
    #             samples_0[index] = samples_0[index][:len(samples_1[index])]
    #
    #         patient_number[pat] = len(samples_0[index]) + len(samples_1[index])
    #
    #         labels_0.append(np.ones(len(samples_0[index])).astype(np.int64))
    #         labels_1.append(np.zeros(len(samples_1[index])).astype(np.int64))
    #
    #     if not os.path.exists(f'number_samples_patient_{dataset}.csv'):
    #         with open(f'number_samples_patient_{dataset}.csv', 'a+', encoding='utf8', newline='') as file:
    #             writer = csv.writer(file)
    #             for key, value in patient_number.items():
    #                 content = [key, value]
    #                 writer.writerow(content)
    #
    #     train_samples, train_labels = [], []
    #     for j in range(len(samples_0)):
    #         # # append preictal
    #         train_samples.append(samples_0[j])
    #         train_labels.append(labels_0[j])
    #
    #         # # append interictal
    #         train_samples.append(samples_1[j])
    #         train_labels.append(labels_1[j])
    #
    #     train_samples = torch.from_numpy(np.concatenate(train_samples))
    #     train_labels = torch.from_numpy(np.concatenate(train_labels))
    #
    #     train_set = data.TensorDataset(train_samples, train_labels)
    #     train_loader = data.DataLoader(
    #         dataset=train_set,
    #         shuffle=True,
    #         batch_size=64,
    #         num_workers=0
    #     )
    #
    #     return train_loader
    #
    # elif len(patients) == 2:
    #     patient_number = {}
    #     with open(f'number_samples_patient_{dataset}.csv') as csvfile:
    #         csv_reader = csv.reader(csvfile)
    #         for row in csv_reader:
    #             patient_number[row[0]] = int(row[1])
    #
    #     # get remaining patients' data
    #     remaining_patients = patients[0]
    #     unlearning_patient = patients[1]
    #     if aug_pertur == 0:
    #         # break the balance rule between the remaining data and unlearning data
    #         unlearning_data_rate = 0.5
    #         # balance the remaining data and unlearning data
    #         unlearning_number = 0.0
    #         for patient in unlearning_patient:
    #             unlearning_number += patient_number[patient]
    #         remaining_data_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
    #                                                                           - unlearning_number)
    #     elif aug_pertur == 1:
    #         unlearning_data_rate = 1.0
    #         unlearning_number = 0.0
    #         for patient in unlearning_patient:
    #             unlearning_number += patient_number[patient]
    #         remaining_data_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
    #                                                                           - unlearning_number)
    #     elif aug_pertur == 2:
    #         unlearning_data_rate = 0.1
    #         unlearning_number = 0.0
    #         for patient in unlearning_patient:
    #             unlearning_number += patient_number[patient]
    #         remaining_data_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
    #                                                                           - unlearning_number)
    #
    #     else:
    #         raise ValueError("aug_pertur expected specified option in '0', '1' and '2', "
    #                          "but got aug_pertur={}".format(aug_pertur))
    #
    #     print("get subset of remaining patients' data")
    #     append_index, len_remaining, len_unlearning = 0, 0, 0
    #     samples_0, labels_0, samples_1, labels_1 = [], [], [], []
    #     for index, pat in enumerate(remaining_patients):  # '10' -> 1, 0
    #         temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
    #         temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()
    #
    #         if dataset == 'CHBMIT':
    #             samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
    #             samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))
    #
    #         elif dataset == 'Kaggle2014Pred':
    #             samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
    #             samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))
    #
    #             np.random.shuffle(samples_0[index])
    #             np.random.shuffle(samples_1[index])
    #
    #         else:
    #             raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
    #                              "but got dataset={}".format(dataset))
    #
    #         # # balance the two classes
    #         if len(samples_1[index]) >= len(samples_0[index]):
    #             np.random.shuffle(samples_1[index])
    #             samples_1[index] = samples_1[index][:len(samples_0[index])]
    #         else:
    #             np.random.shuffle(samples_0[index])
    #             samples_0[index] = samples_0[index][:len(samples_1[index])]
    #
    #         samples_0[index] = samples_0[index][:int(len(samples_0[index]) * remaining_data_rate)]
    #         samples_1[index] = samples_1[index][:int(len(samples_1[index]) * remaining_data_rate)]
    #
    #         # '1' and '0' for preictal and interictal segments of remaining patients, respectively
    #         labels_0.append(np.ones(len(samples_0[index])).astype(np.int64))
    #         labels_1.append(np.zeros(len(samples_1[index])).astype(np.int64))
    #
    #         len_remaining += (len(samples_0[index]) + len(samples_1[index]))
    #
    #         append_index += 1
    #
    #     # get unlearning patients' data
    #     print("\nget unlearning patients' data")
    #     for index, pat in enumerate(unlearning_patient):  # '10' -> 1, 0
    #         temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
    #         temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()
    #
    #         if dataset == 'CHBMIT':
    #             samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
    #             samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))
    #
    #         elif dataset == 'Kaggle2014Pred':
    #             samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
    #             samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))
    #
    #             np.random.shuffle(samples_0[append_index])
    #             np.random.shuffle(samples_1[append_index])
    #
    #         else:
    #             raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
    #                              "but got dataset={}".format(dataset))
    #
    #         # # balance the two classes
    #         if len(samples_1[append_index]) >= len(samples_0[append_index]):
    #             np.random.shuffle(samples_1[append_index])
    #             samples_1[append_index] = samples_1[append_index][:len(samples_0[append_index])]
    #         else:
    #             np.random.shuffle(samples_0[append_index])
    #             samples_0[append_index] = samples_0[append_index][:len(samples_1[append_index])]
    #
    #         # used for 'using part of unlearning patient's data plan'
    #         samples_0[append_index] = samples_0[append_index][:int(len(samples_0[append_index]) * unlearning_data_rate)]
    #         samples_1[append_index] = samples_1[append_index][:int(len(samples_1[append_index]) * unlearning_data_rate)]
    #
    #         # # reverse labels
    #         labels_0.append(np.zeros(len(samples_0[append_index])).astype(np.int64))
    #         labels_1.append(np.ones(len(samples_1[append_index])).astype(np.int64))
    #
    #         len_unlearning += (len(samples_0[append_index]) + len(samples_1[append_index]))
    #
    #         append_index += 1
    #
    #     # key coding
    #     unlearning_samples, unlearning_labels = [], []
    #     for j in range(len(samples_0)):
    #         # # append preictal
    #         unlearning_samples.append(samples_0[j])
    #         unlearning_labels.append(labels_0[j])
    #
    #         # # append interictal
    #         unlearning_samples.append(samples_1[j])
    #         unlearning_labels.append(labels_1[j])
    #     # key coding
    #
    #     unlearning_samples = np.concatenate(unlearning_samples)
    #     unlearning_labels = np.concatenate(unlearning_labels)
    #
    #     unlearning_samples = torch.from_numpy(unlearning_samples)
    #     unlearning_labels = torch.from_numpy(unlearning_labels)
    #
    #     if domain_labels:
    #         # # '0' for samples from remaining patients, '1' for samples from unlearning patient.
    #         unlearning_labels_domain = np.concatenate((np.zeros(len_remaining, dtype=np.int64),
    #                                                    np.ones(len_unlearning, dtype=np.int64)))
    #
    #         unlearning_labels_domain = torch.from_numpy(unlearning_labels_domain)
    #
    #         unlearning_set = data.TensorDataset(unlearning_samples, unlearning_labels, unlearning_labels_domain)
    #         unlearning_loader = data.DataLoader(
    #             dataset=unlearning_set,
    #             shuffle=True,
    #             batch_size=64,
    #             num_workers=0
    #         )
    #
    #         return unlearning_loader
    #
    #     else:
    #         unlearning_set = data.TensorDataset(unlearning_samples, unlearning_labels)
    #         unlearning_loader = data.DataLoader(
    #             dataset=unlearning_set,
    #             shuffle=True,
    #             batch_size=64,
    #             num_workers=0
    #         )
    #
    #         return unlearning_loader

    with open('SETTINGS_%s.json' % dataset) as f:
        settings = json.load(f)

    # data -> set -> loader (for training an original model)
    if len(patients) == 1:
        patient_number = {}
        samples_0, labels_0, samples_1, labels_1 = [], [], [], []
        for index, pat in enumerate(patients[0]):  # '10' -> 1, 0
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            num_preictal = len(temp_pre_s)
            # for some patients the interictal parts' number is different from the preictal parts' number
            if isinstance(temp_inter_s, list):
                temp_inter_s = np.concatenate(temp_inter_s, axis=0)
            inter_folder_len = int(len(temp_inter_s) / num_preictal)
            inter_samples = []
            for i in range(num_preictal):
                inter_samples.append((temp_inter_s[i * inter_folder_len: (i + 1) * inter_folder_len]))
            temp_inter_s = inter_samples

            if dataset == 'CHBMIT':
                temp_pre_s.pop(test_th)
                temp_inter_s.pop(test_th)
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

            elif dataset == 'Kaggle2014Pred':
                temp_th = deepcopy(test_th)
                if pat in ['Dog_3', 'Dog_4']:
                    test_th += 4
                temp_pre_s.pop(test_th)
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

                test_th = temp_th
            else:
                raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                                 "but got dataset={}".format(dataset))

            # # balance the two classes
            if len(samples_1[index]) >= len(samples_0[index]):
                np.random.shuffle(samples_1[index])
                samples_1[index] = samples_1[index][:len(samples_0[index])]
            else:
                np.random.shuffle(samples_0[index])
                samples_0[index] = samples_0[index][:len(samples_1[index])]

            patient_number[pat] = len(samples_0[index]) + len(samples_1[index])

            labels_0.append(np.ones(len(samples_0[index]), dtype=np.int64))
            labels_1.append(np.zeros(len(samples_1[index]), dtype=np.int64))

        if not os.path.exists(f'number_samples_patient_{dataset}.csv'):
            with open(f'number_samples_patient_{dataset}.csv', 'a+', encoding='utf8', newline='') as file:
                writer = csv.writer(file)
                for key, value in patient_number.items():
                    content = [key, value]
                    writer.writerow(content)

        train_samples = np.concatenate((np.concatenate(samples_0), np.concatenate(samples_1)))
        train_labels = np.concatenate((np.concatenate(labels_0), np.concatenate(labels_1)))

        train_samples = torch.from_numpy(train_samples)
        train_labels = torch.from_numpy(train_labels)

        train_set = data.TensorDataset(train_samples, train_labels)
        train_loader = data.DataLoader(
            dataset=train_set,
            shuffle=True,
            batch_size=64,
            num_workers=0
        )

        return train_loader

    elif len(patients) == 2:

        remaining_patients = patients[0]
        unlearning_patient = patients[1]
        # remaining_rate = 0.1
        len_remaining = 0
        len_unlearning = 0

        samples_0, labels_0, samples_1, labels_1 = [], [], [], []
        append_index = 0

        # get number of samples of each patient
        patient_number = {}
        with open(f'number_samples_patient_{dataset}.csv') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                patient_number[row[0]] = int(row[1])

        if aug_pertur == 0:
            # break the balance rule between the remaining data and unlearning data
            unlearning_data_rate = 0.1
            # remaining_rate = 0.1
            # balance the remaining data and unlearning data
            unlearning_number = 0.0
            for patient in unlearning_patient:
                unlearning_number += patient_number[patient]
            remaining_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
                                                                         - unlearning_number)
        elif aug_pertur == 1:
            unlearning_data_rate = 1.0
            unlearning_number = 0.0
            for patient in unlearning_patient:
                unlearning_number += patient_number[patient]
            remaining_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
                                                                         - unlearning_number)
        elif aug_pertur == 2:
            unlearning_data_rate = 0.5
            # balancing unlearning data and remaining data
            remaining_rate = 0.1
            # unlearning_number = 0.0
            # for patient in unlearning_patient:
            #     unlearning_number += patient_number[patient]
            # remaining_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
            #                                                              - unlearning_number)

        else:
            raise ValueError("aug_pertur expected specified option in '0', '1' and '2', "
                             "but got aug_pertur={}".format(aug_pertur))

        # get remaining patients' data
        print("get subset of remaining patients' data")
        for index, pat in enumerate(remaining_patients):  # '10' -> 1, 0
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            num_preictal = len(temp_pre_s)
            # for some patients the interictal parts' number is different from the preictal parts' number
            if isinstance(temp_inter_s, list):
                temp_inter_s = np.concatenate(temp_inter_s, axis=0)
            inter_folder_len = int(len(temp_inter_s) / num_preictal)
            inter_samples = []
            for i in range(num_preictal):
                inter_samples.append((temp_inter_s[i * inter_folder_len: (i + 1) * inter_folder_len]))
            temp_inter_s = inter_samples

            if dataset == 'CHBMIT':
                temp_pre_s.pop(test_th)
                temp_inter_s.pop(test_th)
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

                # # balance the two classes
                if len(samples_1[index]) >= len(samples_0[index]):
                    np.random.shuffle(samples_1[index])
                    samples_1[index] = samples_1[index][:len(samples_0[index])]
                else:
                    np.random.shuffle(samples_0[index])
                    samples_0[index] = samples_0[index][:len(samples_1[index])]

            elif dataset == 'Kaggle2014Pred':
                temp_pre_s.pop(test_th)
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

            else:
                raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                                 "but got dataset={}".format(dataset))

            # # get subset with remaining_rate of each remaining patient
            samples_0[index] = samples_0[index][:int(len(samples_0[index]) * remaining_rate)]
            samples_1[index] = samples_1[index][:int(len(samples_1[index]) * remaining_rate)]
            # # '0' and '1' for preictal and interictal segments of remaining patients, respectively
            labels_0.append(np.ones(len(samples_0[index]), dtype=np.int64))
            labels_1.append(np.zeros(len(samples_1[index]), dtype=np.int64))
            len_remaining += (len(samples_0[index]) + len(samples_1[index]))

            append_index += 1

        # get unlearning patients' data
        if domain_labels:  # for 'only use remaining samples to unlearning method'
            print("\nget unlearning patients' data")
            for index, pat in enumerate(unlearning_patient):  # '10' -> 1, 0
                temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
                temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

                num_preictal = len(temp_pre_s)
                # for some patients the interictal parts' number is different from the preictal parts' number
                if isinstance(temp_inter_s, list):
                    temp_inter_s = np.concatenate(temp_inter_s, axis=0)
                inter_folder_len = int(len(temp_inter_s) / num_preictal)
                inter_samples = []
                for i in range(num_preictal):
                    inter_samples.append((temp_inter_s[i * inter_folder_len: (i + 1) * inter_folder_len]))
                temp_inter_s = inter_samples

                if dataset == 'CHBMIT':
                    temp_pre_s.pop(test_th)
                    temp_inter_s.pop(test_th)
                    samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                    samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

                    # # balance the two classes
                    if len(samples_1[append_index]) >= len(samples_0[append_index]):
                        np.random.shuffle(samples_1[append_index])
                        samples_1[append_index] = samples_1[append_index][:len(samples_0[append_index])]
                    else:
                        np.random.shuffle(samples_0[append_index])
                        samples_0[append_index] = samples_0[append_index][:len(samples_1[append_index])]

                elif dataset == 'Kaggle2014Pred':
                    temp_pre_s.pop(test_th)
                    samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                    samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

                else:
                    raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                                     "but got dataset={}".format(dataset))

                samples_0[append_index] = samples_0[append_index][:int(len(samples_0[append_index]) *
                                                                       unlearning_data_rate)]
                samples_1[append_index] = samples_1[append_index][:int(len(samples_1[append_index]) *
                                                                       unlearning_data_rate)]

                # # # '2' and '3' for preictal and interictal segments of unlearning patient
                # labels_0[append_index] = np.ones(len(samples_0[append_index]), dtype=int) * 2
                # labels_1[append_index] = np.ones(len(samples_1[append_index]), dtype=int) * 3
                # # # random labels
                # labels_0[append_index] = np.random.randint(0, 2, len(samples_0[append_index]), dtype=int)
                # labels_1[append_index] = np.random.randint(0, 2, len(samples_1[append_index]), dtype=int)

                # # reverse labels 1
                labels_0.append(np.zeros(len(samples_0[append_index]), dtype=np.int64))
                labels_1.append(np.ones(len(samples_1[append_index]), dtype=np.int64))

                # # reverse labels 2
                # len_pre = len(samples_0[append_index])
                # len_inter = len(samples_1[append_index])
                # labels_0.append(np.concatenate((np.zeros(int(len_pre/2), dtype=np.int64),
                #                                np.ones(len_pre-int(len_pre/2), dtype=np.int64))))
                # labels_1.append(np.concatenate((np.zeros(len_inter, dtype=np.int64),
                #                                 np.ones(len_inter-int(len_inter/2), dtype=np.int64))))

                len_unlearning += (len(samples_0[append_index]) + len(samples_1[append_index]))

                append_index += 1

        patients_data = {'p_s': samples_0, 'p_l': labels_0, 'i_s': samples_1, 'i_l': labels_1}

        unlearning_samples, unlearning_labels = [], []
        for j in range(len(patients_data['p_s'])):
            # # append preictal
            unlearning_samples.append(patients_data['p_s'][j])
            unlearning_labels.append(patients_data['p_l'][j])

            # # append interictal
            unlearning_samples.append(patients_data['i_s'][j])
            unlearning_labels.append(patients_data['i_l'][j])

        unlearning_samples = np.concatenate(unlearning_samples)
        unlearning_labels = np.concatenate(unlearning_labels)

        unlearning_samples = torch.from_numpy(unlearning_samples)
        unlearning_labels = torch.from_numpy(unlearning_labels)

        if domain_labels:
            # # '0' for samples from remaining patients, '1' for samples from unlearning patient.
            unlearning_labels_domain = np.concatenate((np.zeros(len_remaining, dtype=np.int64),
                                                       np.ones(len_unlearning, dtype=np.int64)))

            unlearning_labels_domain = torch.from_numpy(unlearning_labels_domain)

            unlearning_set = data.TensorDataset(unlearning_samples, unlearning_labels, unlearning_labels_domain)
            unlearning_loader = data.DataLoader(
                dataset=unlearning_set,
                shuffle=True,
                batch_size=64,
                num_workers=0
            )

            return unlearning_loader

        else:
            unlearning_set = data.TensorDataset(unlearning_samples, unlearning_labels)
            unlearning_loader = data.DataLoader(
                dataset=unlearning_set,
                shuffle=True,
                batch_size=64,
                num_workers=0
            )

            return unlearning_loader


def load_data_influence(disturb_methods, dataset, *patients):
    """

    :param disturb_methods: 'samples', 'labels' or 'samples_labels'
    :param patients: [0] remaining patients, [1] forgetting patients
    :param dataset:
    :return: subset loader of remaining patients, loader of forgetting patients and loader of forgetting patients with
             disturb samples or labels
    """
    with open('SETTINGS_%s.json' % dataset) as f:
        settings = json.load(f)

    remaining_rate = 0.1
    remaining_patients = patients[0]
    unlearning_patient = patients[1]

    samples_0, labels_0, samples_1, labels_1 = [], [], [], []
    # get subset of remaining patients' data
    print("get subset of remaining patients' data\n")
    for index, pat in enumerate(remaining_patients):  # '10' -> 1, 0
        temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
        temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

        if dataset == 'CHBMIT':

            samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
            samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

            # # balance the two classes and shuffle
            if len(samples_1[index]) >= len(samples_0[index]):
                np.random.shuffle(samples_1[index])
                samples_1[index] = samples_1[index][:len(samples_0[index])]
            else:
                np.random.shuffle(samples_0[index])
                samples_0[index] = samples_0[index][:len(samples_1[index])]

        elif dataset == 'Kaggle2014Pred':
            samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
            samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))

            np.random.shuffle(samples_0[index])
            np.random.shuffle(samples_1[index])
        else:
            raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                             "but got dataset={}".format(dataset))

        # # get subset with remaining_rate of each remaining patient
        samples_0[index] = samples_0[index][:int(len(samples_0[index]) * remaining_rate)]
        samples_1[index] = samples_1[index][:int(len(samples_1[index]) * remaining_rate)]
        # # '0' and '1' for preictal and interictal segments of remaining patients
        labels_0.append(np.ones(len(samples_0[index])).astype(np.int64))
        labels_1.append(np.zeros(len(samples_1[index])).astype(np.int64))

    remaining_samples = np.concatenate((np.concatenate(samples_0), np.concatenate(samples_1)))
    remaining_labels = np.concatenate((np.concatenate(labels_0), np.concatenate(labels_1)))

    remaining_samples = torch.from_numpy(remaining_samples)
    remaining_labels = torch.from_numpy(remaining_labels)

    remaining_set = data.TensorDataset(remaining_samples, remaining_labels)
    remaining_loader = data.DataLoader(
        dataset=remaining_set,
        shuffle=True,
        batch_size=64,
        num_workers=0
    )

    # get loaders of forgetting patients
    print("\nget forgetting patients' data")
    forget_samples_0, forget_labels_0, forget_samples_1, forget_labels_1 = [], [], [], []
    num = len(unlearning_patient)
    disturb_samples_0, disturb_labels_0, disturb_samples_1, disturb_labels_1 = [] * num, [] * num, [] * num, [] * num
    for index, pat in enumerate(unlearning_patient):  # '10' -> 1, 0
        temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
        temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

        if dataset == 'CHBMIT':
            forget_samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
            forget_samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

            # # balance the two classes
            if len(forget_samples_1[index]) >= len(forget_samples_0[index]):
                np.random.shuffle(forget_samples_1[index])
                forget_samples_1[index] = forget_samples_1[index][:len(forget_samples_0[index])]
            else:
                np.random.shuffle(forget_samples_0[index])
                forget_samples_0[index] = forget_samples_0[index][:len(forget_samples_1[index])]

        elif dataset == 'Kaggle2014Pred':
            samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
            samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))
            # use all data, not shuffle

        else:
            raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                             "but got dataset={}".format(dataset))

        forget_labels_0.append(np.ones(len(forget_samples_0[index])).astype(np.int64))
        forget_labels_1.append(np.zeros(len(forget_samples_1[index])).astype(np.int64))

        # # generate disturb samples or labels
        disturb_samples_0[index], disturb_labels_0[index], disturb_samples_1[index], disturb_labels_1[index] = \
            generate_disturb(forget_samples_0[index], forget_samples_1[index], disturb_methods)

    raw_forgetting_samples = np.concatenate((np.concatenate(forget_samples_0), np.concatenate(forget_samples_1)))
    raw_forgetting_labels = np.concatenate((np.concatenate(forget_labels_0), np.concatenate(forget_labels_1)))
    disturb_forgetting_sampels = np.concatenate((np.concatenate(disturb_samples_0),
                                                 np.concatenate(disturb_samples_1)))
    disturb_forgetting_labels = np.concatenate((np.concatenate(disturb_labels_0),
                                                np.concatenate(disturb_labels_1)))

    raw_forgetting_samples = torch.from_numpy(raw_forgetting_samples)
    raw_forgetting_labels = torch.from_numpy(raw_forgetting_labels)
    disturb_forgetting_sampels = torch.from_numpy(disturb_forgetting_sampels)
    disturb_forgetting_labels = torch.from_numpy(disturb_forgetting_labels)

    raw_forgetting_set = data.TensorDataset(raw_forgetting_samples, raw_forgetting_labels)
    raw_forgetting_loader = data.DataLoader(
        dataset=raw_forgetting_set,
        shuffle=True,
        batch_size=64,
        num_workers=0
    )

    disturb_forgetting_set = data.TensorDataset(disturb_forgetting_sampels, disturb_forgetting_labels)
    disturb_forgetting_loader = data.DataLoader(
        dataset=disturb_forgetting_set,
        shuffle=True,
        batch_size=64,
        num_workers=0
    )

    return remaining_loader, raw_forgetting_loader, disturb_forgetting_loader


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

        # patient_number = {}
        # dataloader_list = []
        # batch_size_settings = {'32': {'Dog_1': 32, 'Dog_2': 32, 'Dog_3': 96, 'Dog_4': 53, 'Dog_5': 29}}
        #
        # for ind, pat in enumerate(patients[0]):
        #     temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
        #     temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()
        #
        #     samples_0 = np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32)
        #     samples_1 = temp_inter_s[:, np.newaxis, :, :].astype(np.float32)
        #
        #     patient_number[pat] = len(samples_0) + len(samples_1)
        #
        #     # # # balance the two classes (useless for kaggle dataset)
        #     # if len(samples_1) >= len(samples_0):
        #     #     np.random.shuffle(samples_1)
        #     #     samples_1 = samples_1[:len(samples_0)]
        #     # else:
        #     #     np.random.shuffle(samples_0)
        #     #     samples_0 = samples_0[:len(samples_1)]
        #
        #     labels_0 = np.ones(len(samples_0)).astype(np.int64)
        #     labels_1 = np.zeros(len(samples_1)).astype(np.int64)
        #
        #     samples_0 = torch.from_numpy(samples_0)
        #     samples_1 = torch.from_numpy(samples_1)
        #     labels_0 = torch.from_numpy(labels_0)
        #     labels_1 = torch.from_numpy(labels_1)
        #
        #     train_samples = torch.cat((samples_0, samples_1))
        #     train_labels = torch.cat((labels_0, labels_1))
        #
        #     train_set = data.TensorDataset(train_samples, train_labels)
        #     train_loader = data.DataLoader(
        #         dataset=train_set,
        #         shuffle=True,
        #         batch_size=batch_size_settings['32'][pat],
        #         num_workers=0
        #     )
        #
        #     dataloader_list.append(train_loader)
        #
        # if not os.path.exists(f'number_samples_patient_{dataset}.csv'):
        #
        #     with open(f'number_samples_patient_{dataset}.csv', 'a+', encoding='utf8', newline='') as file:
        #         writer = csv.writer(file)
        #         for key, value in patient_number.items():
        #             content = [key, value]
        #             writer.writerow(content)
        #
        # return dataloader_list

        patient_number = {}
        samples_0, labels_0, samples_1, labels_1 = [], [], [], []
        for ind, pat in enumerate(patients[0]):
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
            samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))

            patient_number[pat] = len(samples_0[ind]) + len(samples_1[ind])

            labels_0.append(np.ones(len(samples_0[ind])).astype(np.int64))
            labels_1.append(np.zeros(len(samples_1[ind])).astype(np.int64))

        if not os.path.exists(f'number_samples_patient_{dataset}.csv'):

            with open(f'number_samples_patient_{dataset}.csv', 'a+', encoding='utf8', newline='') as file:
                writer = csv.writer(file)
                for key, value in patient_number.items():
                    content = [key, value]
                    writer.writerow(content)

        samples_0 = torch.from_numpy(np.concatenate(samples_0))
        samples_1 = torch.from_numpy(np.concatenate(samples_1))
        labels_0 = torch.from_numpy(np.concatenate(labels_0))
        labels_1 = torch.from_numpy(np.concatenate(labels_1))

        train_samples = torch.cat((samples_0, samples_1))
        train_labels = torch.cat((labels_0, labels_1))

        train_set = data.TensorDataset(train_samples, train_labels)
        train_loader = data.DataLoader(
            dataset=train_set,
            shuffle=True,
            batch_size=128,
            num_workers=0
        )

        return train_loader

    elif len(patients) == 2:  # for training unlearning model

        # for remaining patients
        remaining_patients = patients[0]
        unlearning_patient = patients[1]
        # remaining_rate = 0.1
        len_remaining = 0
        len_unlearning = 0

        samples_0, labels_0, samples_1, labels_1 = [], [], [], []
        append_index = 0

        # get number of samples of each patient
        patient_number = {}
        print('get number of samples of each case')
        with open(f'number_samples_patient_{dataset}.csv') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                patient_number[row[0]] = int(row[1])

        # get remaining patients' data
        if aug_pertur:
            # # in order to get higher performance on remaining data,
            # # break the balance rule between the remaining data and unlearning data
            unlearning_data_rate = 0.1  # used for 'using part of unlearning patient's data plan'
            # # manually set the amount of remaining data
            # remaining_data_rate = 0.1
            # balance the remaining data and unlearning data
            unlearning_number = 0.0
            for patient in unlearning_patient:
                unlearning_number += patient_number[patient]
            remaining_data_rate = unlearning_number * unlearning_data_rate / (sum(patient_number.values())
                                                                              - unlearning_number)
        else:
            unlearning_data_rate = 1.0
            remaining_data_rate = 0.1

        print("get subset of remaining patients' data")
        for index, pat in enumerate(remaining_patients):
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
            samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))

            np.random.shuffle(samples_0[index])
            np.random.shuffle(samples_1[index])

            samples_0[index] = samples_0[index][:int(len(samples_0[index]) * remaining_data_rate)]
            samples_1[index] = samples_1[index][:int(len(samples_1[index]) * remaining_data_rate)]
            # # '0' and '1' for preictal and interictal segments of remaining patients, respectively
            labels_0.append(np.ones(len(samples_0[index])).astype(np.int64))
            labels_1.append(np.zeros(len(samples_1[index])).astype(np.int64))
            len_remaining += (len(samples_0[index]) + len(samples_1[index]))

            append_index += 1

        # get unlearning patients' data
        print("\nget unlearning patients' data")
        for index, pat in enumerate(unlearning_patient):
            temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
            temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

            samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
            samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))

            if aug_pertur:
                np.random.shuffle(samples_0[append_index])
                np.random.shuffle(samples_1[append_index])
            else:
                pass

            # used for 'using part of unlearning patient's data plan'
            samples_0[append_index] = samples_0[append_index][:int(len(samples_0[append_index]) * unlearning_data_rate)]
            samples_1[append_index] = samples_1[append_index][:int(len(samples_1[append_index]) * unlearning_data_rate)]

            labels_0.append(np.zeros(len(samples_0[append_index])).astype(np.int64))
            labels_1.append(np.ones(len(samples_1[append_index])).astype(np.int64))
            len_unlearning += (len(samples_0[append_index]) + len(samples_1[append_index]))

            append_index += 1

        unlearning_samples = torch.from_numpy(np.concatenate((np.concatenate(samples_0), np.concatenate(samples_1))))
        unlearning_labels = torch.from_numpy(np.concatenate((np.concatenate(labels_0), np.concatenate(labels_1))))

        if domain_labels:
            # # '0' for samples from remaining patients, '1' for samples from unlearning patient.
            unlearning_labels_domain = np.concatenate((np.zeros(len_remaining, dtype=np.int64),
                                                       np.ones(len_unlearning, dtype=np.int64)))

            unlearning_labels_domain = torch.from_numpy(unlearning_labels_domain)

            unlearning_set = data.TensorDataset(unlearning_samples, unlearning_labels, unlearning_labels_domain)
            unlearning_loader = data.DataLoader(
                dataset=unlearning_set,
                shuffle=True,
                batch_size=64,
                num_workers=0
            )

            return unlearning_loader

        else:
            unlearning_set = data.TensorDataset(unlearning_samples, unlearning_labels)
            unlearning_loader = data.DataLoader(
                dataset=unlearning_set,
                shuffle=True,
                batch_size=64,
                num_workers=0
            )

            return unlearning_loader
    else:
        print('unexpected argument patients')
        exit()


def generate_disturb(raw_samples_pre, raw_samples_inter, disturb_methods):
    """

    :param raw_samples_pre: data shape: [N, 1024, 22]
    :param raw_samples_inter: data shape: [N, 1024, 22]
    :param disturb_methods: 1): samples: randomly assign values to pre and inter samples with a range from min to max
                            2): labels: reverse labels or randomly assign labels (0, 1)
                            3): samples and labels
    :return:
    """
    # change samples
    if disturb_methods == 'samples':
        print("disturb methods: samples")
        n_pre, l_pre, c_pre = raw_samples_pre.shape
        n_inter, l_inter, c_inter = raw_samples_inter.shape

        print('randomly assign values')
        # # # randomly assign min to max values
        # samples_pre_min, samples_pre_max = raw_samples_pre[0].min(), raw_samples_pre[0].max()
        # samples_inter_min, samples_inter_max = raw_samples_inter[0].min(), raw_samples_inter[0].max()
        # disturb_pre_samples = np.random.uniform(samples_pre_min, samples_pre_max,
        #                                         (n_pre, l_pre, c_pre)).astype(np.float32)
        # disturb_inter_samples = np.random.uniform(samples_inter_min, samples_inter_max,
        #                                           (n_inter, l_inter, c_inter)).astype(np.float32)

        # # randomly assign 0 to 1 values
        disturb_pre_samples = np.random.uniform(0, 1, (n_pre, l_pre, c_pre)).astype(np.float32)
        disturb_inter_samples = np.random.uniform(0, 1, (n_inter, l_inter, c_inter)).astype(np.float32)

        pre_labels = np.ones(n_pre, dtype=int)
        inter_labels = np.zeros(n_inter, dtype=int)

        # print('add noise')
        # noise_pre = np.random.uniform(-100, 100, (n_pre, l_pre, c_pre)).astype(np.float32)
        # noise_inter = np.random.uniform(-100, 100, (n_inter, l_inter, c_inter)).astype(np.float32)
        # disturb_pre_samples = raw_samples_pre + noise_pre
        # disturb_inter_samples = raw_samples_inter + noise_inter
        #
        # pre_labels = np.ones(n_pre, dtype=int)
        # inter_labels = np.zeros(n_inter, dtype=int)

        # print('add small noise')  # may lead small changes in models
        # noise_pre = np.random.rand(n_pre, l_pre, c_pre).astype(np.float32)
        # noise_inter = np.random.rand(n_inter, l_inter, c_inter).astype(np.float32)
        # disturb_pre_samples = raw_samples_pre + noise_pre
        # disturb_inter_samples = raw_samples_inter + noise_inter
        #
        # pre_labels = np.ones(n_pre, dtype=int)
        # inter_labels = np.zeros(n_inter, dtype=int)

        disturb_pre_samples = disturb_pre_samples[:, np.newaxis, :, :].astype(np.float32)
        disturb_inter_samples = disturb_inter_samples[:, np.newaxis, :, :].astype(np.float32)
        pre_labels = pre_labels.astype(np.int64)
        inter_labels = inter_labels.astype(np.int64)

        return disturb_pre_samples, pre_labels, disturb_inter_samples, inter_labels

    # change labels
    elif disturb_methods == 'labels':
        print("disturb methods: labels")
        # # # reverse labels
        # print('reverse labels')
        # disturb_pre_labels = np.zeros(len(raw_samples_pre), dtype=np.int64)
        # disturb_inter_labels = np.ones(len(raw_samples_inter), dtype=np.int64)

        # # random labels
        print('random labels')
        disturb_pre_labels = np.random.randint(0, 2, len(raw_samples_pre), dtype=np.int64)
        disturb_inter_labels = np.random.randint(0, 2, len(raw_samples_inter), dtype=np.int64)

        raw_samples_pre = raw_samples_pre[:, np.newaxis, :, :].astype(np.float32)
        raw_samples_pre = raw_samples_pre[:, np.newaxis, :, :].astype(np.float32)

        return raw_samples_pre, disturb_pre_labels, raw_samples_inter, disturb_inter_labels

    # change samples and labels
    elif disturb_methods == 'samples_labels':
        print("disturb methods: samples and labels")
        # # random values for each sample
        samples_pre_min, samples_pre_max = raw_samples_pre[0].min(), raw_samples_pre[0].max()
        n_pre, l_pre, c_pre = raw_samples_pre.shape
        samples_inter_min, samples_inter_max = raw_samples_inter[0].min(), raw_samples_inter[0].max()
        n_inter, l_inter, c_inter = raw_samples_inter.shape

        disturb_pre_samples = np.random.uniform(samples_pre_min, samples_pre_max,
                                                (n_pre, l_pre, c_pre)).astype(np.float32)
        disturb_inter_samples = np.random.uniform(samples_inter_min, samples_inter_max,
                                                  (n_inter, l_inter, c_inter)).astype(np.float32)

        # # random labels
        disturb_pre_labels = np.random.randint(0, 2, len(raw_samples_pre), dtype=np.int64)
        disturb_inter_labels = np.random.randint(0, 2, len(raw_samples_inter), dtype=np.int64)

        disturb_pre_samples = disturb_pre_samples[:, np.newaxis, :, :]
        disturb_inter_samples = disturb_inter_samples[:, np.newaxis, :, :]

        return disturb_pre_samples, disturb_pre_labels, disturb_inter_samples, disturb_inter_labels

    else:
        print('unexcept distub methods')
        exit(0)


def test_metrics_records(dataloader_list, model, device, test_th, model_type):
    """
    testing on adaptation data and remain data, and giving metrixs of Sn, SEN, AUC and FPR and statistical analysis of
    p-value.
    :param dataloader_list: consist of dataloader which represents one seizure data (one preictal and one interictal)
    :param model: retrained model
    :param device:
    :param test_th:
    :param model_type:
    :return: None
    """
    feature_records = []
    feature_labels = []

    sensitivity, specificity, auc, fpr, f1_score, precision, acc = [], [], [], [], [], [], []
    corrects = 0
    num_samples = 0
    index = 0
    for dataloader in dataloader_list:
        outs, preds, labels = [], [], []
        samples = []
        model.to(device)
        model.eval()
        with torch.no_grad():
            for sample, label in dataloader:

                sample = sample.to(device)
                label = label.to(device)

                out = model(sample)

                feature_records.append(model.featurer(sample))
                feature_labels.append(label)

                if not np.isfinite(sample.cpu().numpy().any()):
                    print('bad sampels')

                _, pred = torch.max(out, dim=1)
                corrects += torch.eq(pred, label).sum().item()
                num_samples += label.shape[0]
                samples.append(sample)
                outs.append(func.softmax(out, dim=1))
                preds.append(pred)
                labels.append(label)

        outs = torch.cat(outs, dim=0)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # calculate Sn, SEN, AUC, FPR
        outs = outs.cpu().numpy()
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        boundary = np.where(labels == 0)[0][0]

        tp = np.equal(preds[0:boundary], labels[0:boundary]).sum()
        fn = boundary - tp
        tn = np.equal(preds[boundary:], labels[boundary:]).sum()
        fp = len(labels) - boundary - tn
        sensitivity.append(tp / (tp + fn))
        specificity.append((tn / (tn + fp)))
        fpr.append(1 - specificity[-1])
        # below two lines could warn when running
        if tp == 0 and fp == 0:
            precision.append(0)
        else:
            precision.append(tp / (tp + fp))
        if tp == 0:
            f1_score.append(0)
        else:
            f1_score.append(2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))))
        acc.append((tp + tn) / len(labels))
        outs_p = np.concatenate(outs)[1::2]

        try:
            auc.append(metrics.roc_auc_score(labels, outs_p))
        except ValueError:
            for i, t in enumerate(outs_p):
                if not np.isfinite(t):
                    outs_p[i] = 1e-6
            auc.append(metrics.roc_auc_score(labels, outs_p))

        index += 1

    metrics_ = [sensitivity, specificity, precision, acc, f1_score, auc, fpr]

    feature_records = torch.cat(feature_records)
    feature_records = feature_records.cpu().numpy()
    feature_labels = torch.cat(feature_labels)
    feature_labels = feature_labels.cpu().numpy().reshape(-1, 1)
    features = np.concatenate((feature_records, feature_labels), axis=1)
    if model_type == 'original':
        save_path = f'results/feature records/original model'
        os.makedirs(save_path, exist_ok=True)
    elif model_type == 'unlearn':
        save_path = f'results/feature records/unlearned model'
        os.makedirs(save_path, exist_ok=True)
    else:
        raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                         "but got dataset={}".format(model_type))

    features = pd.DataFrame(features)
    features.to_csv(os.path.join(save_path, f'feature distribution_{test_th}.csv'))

    return metrics_


def test_metrics_records_retrain(dataloader_list, model, device):
    """
    testing on adaptation data and remain data, and giving metrixs of Sn, SEN, AUC and FPR and statistical analysis of
    p-value.
    :param dataloader_list: consist of dataloader which represents one seizure data (one preictal and one interictal)
    :param model: retrained model
    :param device:
    :return: None
    """

    sensitivity, specificity, auc, fpr, f1_score, precision, acc = [], [], [], [], [], [], []
    corrects = 0
    num_samples = 0
    index = 0

    len_dataloader = len(dataloader_list)
    last_five = []
    for i in range(5):
        last_five.append(len_dataloader-1-i)

    for dataloader in dataloader_list:
        outs, preds, labels = [], [], []
        samples = []
        model.to(device)
        model.eval()
        feature_records = []
        feature_labels = []
        with torch.no_grad():
            for sample, label in dataloader:

                sample = sample.to(device)
                label = label.to(device)

                out = model(sample)
                if index in last_five:
                    feature_records.append(model.featurer(sample))
                    feature_labels.append(label)

                if not np.isfinite(sample.cpu().numpy().any()):
                    print('bad sampels')

                _, pred = torch.max(out, dim=1)
                corrects += torch.eq(pred, label).sum().item()
                num_samples += label.shape[0]
                samples.append(sample)
                outs.append(func.softmax(out, dim=1))
                preds.append(pred)
                labels.append(label)

        outs = torch.cat(outs, dim=0)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # calculate Sn, SEN, AUC, FPR
        outs = outs.cpu().numpy()
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        boundary = np.where(labels == 0)[0][0]

        tp = np.equal(preds[0:boundary], labels[0:boundary]).sum()
        fn = boundary - tp
        tn = np.equal(preds[boundary:], labels[boundary:]).sum()
        fp = len(labels) - boundary - tn
        sensitivity.append(tp / (tp + fn))
        specificity.append((tn / (tn + fp)))
        fpr.append(1 - specificity[-1])
        # below two lines could warn when running
        if tp == 0 and fp == 0:
            precision.append(0)
        else:
            precision.append(tp / (tp + fp))
        if tp == 0:
            f1_score.append(0)
        else:
            f1_score.append(2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))))
        acc.append((tp + tn) / len(labels))
        outs_p = np.concatenate(outs)[1::2]

        try:
            auc.append(metrics.roc_auc_score(labels, outs_p))
        except ValueError:
            for i, t in enumerate(outs_p):
                if not np.isfinite(t):
                    outs_p[i] = 1e-6
            auc.append(metrics.roc_auc_score(labels, outs_p))

        if index in last_five:
            feature_records = torch.cat(feature_records)
            feature_records = feature_records.cpu().numpy()
            feature_labels = torch.cat(feature_labels)
            feature_labels = feature_labels.cpu().numpy().reshape(-1, 1)
            features = np.concatenate((feature_records, feature_labels), axis=1)
            save_path = f'results/feature records/retrained model'
            os.makedirs(save_path, exist_ok=True)

            features = pd.DataFrame(features)
            features.to_csv(os.path.join(save_path, f'feature distribution_{index}.csv'))

        index += 1

    metrics_ = [sensitivity, specificity, precision, acc, f1_score, auc, fpr]

    return metrics_


def test_metrics(dataloader_list, model, device):
    """
    testing on adaptation data and remain data, and giving metrixs of Sn, SEN, AUC and FPR and statistical analysis of
    p-value.
    :param dataloader_list: consist of dataloader which represents one seizure data (one preictal and one interictal)
    :param model: retrained model
    :param device:
    :return: None
    """

    sensitivity, specificity, auc, fpr, f1_score, precision, acc = [], [], [], [], [], [], []
    corrects = 0
    num_samples = 0
    index = 0
    for dataloader in dataloader_list:
        outs, preds, labels = [], [], []
        samples = []
        model.to(device)
        model.eval()
        with torch.no_grad():
            for sample, label in dataloader:

                sample = sample.to(device)
                label = label.to(device)

                out = model(sample)

                if not np.isfinite(sample.cpu().numpy().any()):
                    print('bad sampels')

                _, pred = torch.max(out, dim=1)
                corrects += torch.eq(pred, label).sum().item()
                num_samples += label.shape[0]
                samples.append(sample)
                outs.append(func.softmax(out, dim=1))
                preds.append(pred)
                labels.append(label)

        outs = torch.cat(outs, dim=0)
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # calculate Sn, SEN, AUC, FPR
        outs = outs.cpu().numpy()
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        boundary = np.where(labels == 0)[0][0]

        tp = np.equal(preds[0:boundary], labels[0:boundary]).sum()
        fn = boundary - tp
        tn = np.equal(preds[boundary:], labels[boundary:]).sum()
        fp = len(labels) - boundary - tn
        sensitivity.append(tp / (tp + fn))
        specificity.append((tn / (tn + fp)))
        fpr.append(1 - specificity[-1])
        # below two lines could warn when running
        if tp == 0 and fp == 0:
            precision.append(0)
        else:
            precision.append(tp / (tp + fp))
        if tp == 0:
            f1_score.append(0)
        else:
            f1_score.append(2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))))
        acc.append((tp + tn) / len(labels))
        outs_p = np.concatenate(outs)[1::2]

        try:
            auc.append(metrics.roc_auc_score(labels, outs_p))
        except ValueError:
            for i, t in enumerate(outs_p):
                if not np.isfinite(t):
                    outs_p[i] = 1e-6
            auc.append(metrics.roc_auc_score(labels, outs_p))

        index += 1

    metrics_ = [sensitivity, specificity, precision, acc, f1_score, auc, fpr]

    return metrics_


def prep_test1(dataset, domain):
    """
    get one patient data, and make it to loader
    :param dataset:
    :param domain: str in a list, contant one patient, like ['1']
    :return: dataloader list
    """
    with open('SETTINGS_%s.json' % dataset) as f:
        settings = json.load(f)

    temp_pre_s, temp_pre_l = PrepData(domain[0], type='ictal', settings=settings).apply()
    temp_inter_s, temp_inter_l = PrepData(domain[0], type='interictal', settings=settings).apply()

    num_preictal = len(temp_pre_s)
    # for some patients the interictal parts' number is different from the preictal parts' number
    if isinstance(temp_inter_s, list):
        temp_inter_s = np.concatenate(temp_inter_s, axis=0)
        temp_inter_l = np.concatenate(temp_inter_l, axis=0)
    inter_folder_len = int(len(temp_inter_s) / num_preictal)
    inter_samples = []
    inter_labels = []
    for i in range(num_preictal):
        inter_samples.append((temp_inter_s[i * inter_folder_len: (i + 1) * inter_folder_len]))
        inter_labels.append((temp_inter_l[i * inter_folder_len: (i + 1) * inter_folder_len]))
    temp_inter_s = inter_samples
    temp_inter_l = inter_labels

    dataloader_list = []
    for i in range(len(temp_pre_s)):

        temp_s = np.concatenate((temp_pre_s[i], temp_inter_s[i]))
        temp_l = np.concatenate((temp_pre_l[i], temp_inter_l[i]))

        temp_s = temp_s[:, np.newaxis, :, :]
        temp_l[temp_l == 2] = 1

        temp_s = temp_s.astype(np.float32)
        temp_l = temp_l.astype(np.int64)

        temp_s = torch.from_numpy(temp_s)
        temp_l = torch.from_numpy(temp_l)

        dataset = data.TensorDataset(temp_s, temp_l)
        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
        )

        dataloader_list.append(dataloader)

    return dataloader_list


def prep_test1_last_seizure_as_test(dataset, domain, test_th):
    """
    get one patient data, and make it to loader
    :param dataset:
    :param domain: str in a list, contant one patient, like ['1']
    :param test_th: the index of test seizure data
    :return: dataloader list
    """
    with open('SETTINGS_%s.json' % dataset) as f:
        settings = json.load(f)

    temp_pre_s, temp_pre_l = PrepData(domain[0], type='ictal', settings=settings).apply()
    temp_inter_s, temp_inter_l = PrepData(domain[0], type='interictal', settings=settings).apply()

    num_preictal = len(temp_pre_s)
    # for some patients the interictal parts' number is different from the preictal parts' number
    if isinstance(temp_inter_s, list):
        temp_inter_s = np.concatenate(temp_inter_s, axis=0)
        temp_inter_l = np.concatenate(temp_inter_l, axis=0)
    inter_folder_len = int(len(temp_inter_s) / num_preictal)
    inter_samples = []
    inter_labels = []
    for i in range(num_preictal):
        inter_samples.append((temp_inter_s[i * inter_folder_len: (i + 1) * inter_folder_len]))
        inter_labels.append((temp_inter_l[i * inter_folder_len: (i + 1) * inter_folder_len]))
    temp_inter_s = inter_samples
    temp_inter_l = inter_labels

    dataloader_list = []

    temp_s = np.concatenate((temp_pre_s[test_th], temp_inter_s[test_th]))
    temp_l = np.concatenate((temp_pre_l[test_th], temp_inter_l[test_th]))

    temp_s = temp_s[:, np.newaxis, :, :]
    temp_l[temp_l == 2] = 1

    temp_s = temp_s.astype(np.float32)
    temp_l = temp_l.astype(np.int64)

    temp_s = torch.from_numpy(temp_s)
    temp_l = torch.from_numpy(temp_l)

    dataset = data.TensorDataset(temp_s, temp_l)
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    dataloader_list.append(dataloader)

    return dataloader_list


def prep_test_kaggle(domain):
    """
    get one patient data, and make it to loader
    :param domain: str in a list, contant one patient, like ['1']
    :return: dataloader list
    """
    dataset = 'Kaggle2014Pred'
    with open('SETTINGS_%s.json' % dataset) as f:
        settings = json.load(f)

    temp_pre_s, temp_pre_l = PrepData(domain[0], type='ictal', settings=settings).apply()
    temp_inter_s, temp_inter_l = PrepData(domain[0], type='interictal', settings=settings).apply()

    num_preictal = len(temp_pre_s)
    inter_folder_len = int(len(temp_inter_s) / num_preictal)
    inter_samples = []
    inter_labels = []
    for i in range(num_preictal):
        inter_samples.append((temp_inter_s[i * inter_folder_len: (i + 1) * inter_folder_len]))
        inter_labels.append((temp_inter_l[i * inter_folder_len: (i + 1) * inter_folder_len]))
    temp_inter_s = inter_samples
    temp_inter_l = inter_labels

    dataloader_list = []
    for i in range(len(temp_pre_s)):

        temp_s = np.concatenate((temp_pre_s[i], temp_inter_s[i]))
        temp_l = np.concatenate((temp_pre_l[i], temp_inter_l[i]))

        temp_s = temp_s[:, np.newaxis, :, :]
        temp_l[temp_l == 2] = 1

        temp_s = temp_s.astype(np.float32)
        temp_l = temp_l.astype(np.int64)

        temp_s = torch.from_numpy(temp_s)
        temp_l = torch.from_numpy(temp_l)

        dataset = data.TensorDataset(temp_s, temp_l)
        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
        )

        dataloader_list.append(dataloader)

    return dataloader_list


def obtain_attack_models(dataset, test_th, patients, raw_model, device, attack_model_path):

    # load data for membership inference attack
    shadow_s_l = load_data_mia(dataset, test_th, patients)
    shadow_samples, shadow_labels = shadow_s_l[0], shadow_s_l[1]

    # train shadow models
    # # split shadow data into train and split dataset
    train_test_rate = 0.5  # according to the paper 'Membership Inference Attacks against Machine Learning Models'
    num_shadow_model = 3
    in_predictions, in_labels, out_predictions, out_labels = [], [], [], []
    true_labels_in, true_labels_out = [], []  # 'in': 1; 'out': 0
    for i in range(num_shadow_model):  # train shadow models for 3 times (my default)
        # # disrupt data
        random_index = np.random.permutation(len(shadow_samples))
        shadow_samples = shadow_samples[random_index]
        shadow_labels = shadow_labels[random_index]

        # # split shadow dataset into train and test dataset
        shadow_train_s = shadow_samples[:int(len(shadow_samples) * train_test_rate)]
        shadow_train_l = shadow_labels[:int(len(shadow_labels) * train_test_rate)]
        shadow_test_s = shadow_samples[int(len(shadow_samples) * train_test_rate):]
        shadow_test_l = shadow_labels[int(len(shadow_labels) * train_test_rate):]

        shadow_train_s = torch.from_numpy(shadow_train_s)
        shadow_train_l = torch.from_numpy(shadow_train_l)
        shadow_test_s = torch.from_numpy(shadow_test_s)
        shadow_test_l = torch.from_numpy(shadow_test_l)

        shadow_train_set = data.TensorDataset(shadow_train_s, shadow_train_l)
        shadow_train_loader = data.DataLoader(
            dataset=shadow_train_set,
            batch_size=64,
            shuffle=True,
            num_workers=0
        )

        shadow_test_set = data.TensorDataset(shadow_test_s, shadow_test_l)
        shadow_test_loader = data.DataLoader(
            dataset=shadow_test_set,
            batch_size=64,
            shuffle=False,
            num_workers=0
        )

        # # train shadow models
        shadow_model = deepcopy(raw_model)
        optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()

        epochs = 5
        print('------------train shadow models------------')
        for epoch in range(epochs):
            train_losses = []
            shadow_model = shadow_model.to(device)
            shadow_model.train()
            loop = tqdm(shadow_train_loader, ascii=True)
            for samples, labels in loop:
                samples = samples.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                class_preds = shadow_model(samples)
                loss = criterion(class_preds, labels)

                loss.backward()

                optimizer.step()

                train_losses.append(loss.item() / labels.shape[0])
                loop.set_description(
                    f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(train_losses) / len(train_losses)):.10f}"
                )

            loop.close()

        in_predictions_th = []  # 在训练集中的样本的预测值
        true_labels_in_th = []  # 在训练集中的样本的pre和inter标签
        out_predictions_th = []
        true_labels_out_th = []

        with torch.no_grad():
            shadow_model.to(device)
            shadow_model.eval()
            # # get 'in' predictions
            for samples, labels in shadow_train_loader:
                samples = samples.to(device)
                true_labels_in_th.append(labels)

                predictions = shadow_model(samples)
                in_predictions_th.append(func.softmax(predictions, dim=-1))

            in_predictions_th = torch.cat(in_predictions_th).cpu().numpy().astype(np.float32)
            true_labels_in_th = torch.cat(true_labels_in_th).numpy().astype(np.int64)
            in_labels_th = np.ones(len(in_predictions_th), dtype=np.int64)

            in_predictions.append(in_predictions_th)
            true_labels_in.append(true_labels_in_th)
            in_labels.append(in_labels_th)

            # # get 'out' predictions
            for samples, labels in shadow_test_loader:
                samples = samples.to(device)
                true_labels_out_th.append(labels)

                predictions = shadow_model(samples)
                out_predictions_th.append(func.softmax(predictions, dim=-1))

            out_predictions_th = torch.cat(out_predictions_th).cpu().numpy().astype(np.float32)
            true_labels_out_th = torch.cat(true_labels_out_th).numpy().astype(np.int64)
            out_labels_th = np.zeros(len(out_predictions_th), dtype=np.int64)

            out_predictions.append(out_predictions_th)
            true_labels_out.append(true_labels_out_th)
            out_labels.append(out_labels_th)

    # train attack model
    # # make predictions and true labels into samples, 'in' and 'out' labels into labels for training attack models
    num_attack_models = np.unique(true_labels_in)

    attack_samples = np.concatenate((np.concatenate(in_predictions), np.concatenate(out_predictions)))
    attack_labels = np.concatenate((np.concatenate(in_labels), np.concatenate(out_labels)))
    true_labels = np.concatenate((np.concatenate(true_labels_in), np.concatenate(true_labels_out)))
    indices = np.arange(len(true_labels))

    attack_models = []
    for i in num_attack_models:
        attack_s_l_i_indices = indices[true_labels == i]
        attack_samples_i = attack_samples[attack_s_l_i_indices]
        attack_labels_i = attack_labels[attack_s_l_i_indices]

        attack_samples_i = torch.from_numpy(attack_samples_i)
        attack_labels_i = torch.from_numpy(attack_labels_i)

        attack_set = data.TensorDataset(attack_samples_i, attack_labels_i)
        attack_loader = data.DataLoader(
            dataset=attack_set,
            batch_size=64,
            shuffle=True,
            num_workers=0
        )

        # # construct attack model
        attack_model = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(2, 4)),
            ('acti_1', nn.ReLU()),
            ('linear_2', nn.Linear(4, 8)),
            ('acti_2', nn.ReLU()),
            ('linear_3', nn.Linear(8, 4)),
            ('acti_3', nn.ReLU()),
            ('linear_4', nn.Linear(4, 2)),
        ]))

        optimizer = optim.Adam(attack_model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()

        epochs = 5
        print(f'------------train {i+1} th attack models------------')
        for epoch in range(epochs):
            train_losses = []
            attack_model.to(device)
            attack_model.train()
            loop = tqdm(attack_loader, ascii=True)
            for samples, labels in loop:
                samples = samples.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                class_preds = attack_model(samples)
                loss = criterion(class_preds, labels)

                loss.backward()

                optimizer.step()

                train_losses.append(loss.item() / labels.shape[0])
                loop.set_description(
                    f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(train_losses) / len(train_losses)):.10f}"
                )

            loop.close()

        attack_models.append(attack_model)

        torch.save(attack_model, os.path.join(attack_model_path,
                                              f'test_th_{test_th}_category_{i + 1}_attack_model.pth'))
    return attack_models


def load_data_mia(dataset, test_th, patients):
    """
    get patients' data and obtain a dataloader by using the data
    :param patients: must str in list.
                    1) one variable: original patients
                    2) two variables: remaining patients and unleanring patient
    :param dataset:
    :param test_th: the index of testing seizure data
    :return: 1) one variable: return a dataloader used for training original model
             2) two variables: return a dataloader used for training unlearning model
    """

    with open('SETTINGS_%s.json' % dataset) as f:
        settings = json.load(f)

    # data -> set -> loader (for training an original model)
    shadow_samples_0, shadow_labels_0, shadow_samples_1, shadow_labels_1 = [], [], [], []
    for index, pat in enumerate(patients):  # '10' -> 1, 0
        temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
        temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

        num_preictal = len(temp_pre_s)
        # for some patients the interictal parts' number is different from the preictal parts' number
        if isinstance(temp_inter_s, list):
            temp_inter_s = np.concatenate(temp_inter_s, axis=0)
        inter_folder_len = int(len(temp_inter_s) / num_preictal)
        inter_samples = []
        for i in range(num_preictal):
            inter_samples.append((temp_inter_s[i * inter_folder_len: (i + 1) * inter_folder_len]))
        temp_inter_s = inter_samples

        if dataset == 'CHBMIT':
            shadow_samples_0.append(temp_pre_s.pop(test_th)[:, np.newaxis, :, :].astype(np.float32))
            shadow_samples_1.append(temp_inter_s.pop(test_th)[:, np.newaxis, :, :].astype(np.float32))

        elif dataset == 'Kaggle2014Pred':

            shadow_samples_0.append(temp_pre_s.pop(test_th)[:, np.newaxis, :, :].astype(np.float32))
            shadow_samples_1.append(temp_inter_s.pop(test_th)[:, np.newaxis, :, :].astype(np.float32))

        else:
            raise ValueError("dataset expected specified option in 'CHBMIT', 'Kaggle2014Pred'"
                             "but got dataset={}".format(dataset))

        shadow_labels_0.append(np.ones(len(shadow_samples_0[index]), dtype=np.int64))
        shadow_labels_1.append(np.zeros(len(shadow_samples_1[index]), dtype=np.int64))

    shadow_s_l = [np.concatenate((np.concatenate(shadow_samples_0), np.concatenate(shadow_samples_1))),
                  np.concatenate((np.concatenate(shadow_labels_0), np.concatenate(shadow_labels_1)))]

    return shadow_s_l
