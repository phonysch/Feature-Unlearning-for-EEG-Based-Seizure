from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import torch
from unlearning_scheme_functions import separate_prediction, loss_Tsallis
from adv_generator import FGSM
import os
from PCT_net import PCT
from CNN import CNN_kaggle
from CNN import CNN
from tqdm import tqdm
from Transformer import ViT
from Transformer import ViT_kaggle
from STMLP_configs import configs
import json
import numpy as np
import torch.utils.data as data
from load_data import PrepData
from functions import obtain_attack_models
from RepNet import RepNet


def membership_inference_attack(dataset, test_th, unlearning_patients, method, target_model, attack_models, device):
    # # obtain attack success rate (ASR)
    # # # obtain unlearning data
    pop = True
    pre_loader, inter_loader = load_data_balance_attack(dataset, test_th, unlearning_patients, method, pop)

    target_model.to(device)
    target_model.eval()
    pre_preds, in_out_labels_0, inter_preds, in_out_labels_1 = [], [], [], []
    with torch.no_grad():

        # # # obtain target model predictions
        for samples, in_out_labels in pre_loader:  # labels: 1
            samples = samples.to(device)
            predictions = func.softmax(target_model(samples), dim=-1)
            pre_preds.append(predictions)
            in_out_labels_0.append(in_out_labels)

        for samples, in_out_labels in inter_loader:  # labels: 0
            samples = samples.to(device)
            predictions = func.softmax(target_model(samples), dim=-1)
            inter_preds.append(predictions)
            in_out_labels_1.append(in_out_labels)

        pre_preds, inter_preds = torch.cat(pre_preds).cpu(), torch.cat(inter_preds).cpu()
        in_out_labels_0, in_out_labels_1 = torch.cat(in_out_labels_0), torch.cat(in_out_labels_1)

        pre_preds_set = data.TensorDataset(pre_preds, in_out_labels_0)
        pre_preds_loader = data.DataLoader(
            dataset=pre_preds_set,
            shuffle=False,
            batch_size=1024,
            num_workers=0
        )

        inter_preds_set = data.TensorDataset(inter_preds, in_out_labels_1)
        inter_preds_loader = data.DataLoader(
            dataset=inter_preds_set,
            shuffle=False,
            batch_size=1024,
            num_workers=0
        )

        # # # obtain ASR on pre and inter data seperately
        corrects, num_samples = 0, 0
        for samples, labels in inter_preds_loader:
            samples = samples.to(device)
            labels = labels.to(device)
            predictions = attack_models[0](samples)

            _, preds = torch.max(predictions, dim=1)
            corrects += torch.eq(preds, labels).sum().item()
            num_samples += labels.shape[0]

        for samples, labels in pre_preds_loader:
            samples = samples.to(device)
            labels = labels.to(device)
            predictions = attack_models[1](samples)

            _, preds = torch.max(predictions, dim=1)
            corrects += torch.eq(preds, labels).sum().item()
            num_samples += labels.shape[0]

        print(f'attack success rate: {(corrects / num_samples):.3f} ------ patient: {unlearning_patients[0]}')


def load_data_balance_attack(dataset, test_th, patients, method, pop):

    with open('SETTINGS_%s.json' % dataset) as f:
        settings = json.load(f)

    # data -> set -> loader (for training a original model)
    samples_0, samples_1 = [], []
    in_out_labels_0, in_out_labels_1 = [], []
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

        if not pop:
            if dataset == 'CHBMIT':
                temp_pre_s.pop(test_th)
                temp_inter_s.pop(test_th)
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

            elif dataset == 'Kaggle2014Pred':
                # temp_th = deepcopy(test_th)
                # if pat in ['Dog_3', 'Dog_4']:
                #     test_th += 4
                temp_pre_s.pop(test_th)
                temp_inter_s.pop(test_th)
                samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

                # test_th = temp_th
            else:
                raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                                 "but got dataset={}".format(dataset))

        else:
            if dataset == 'CHBMIT':

                samples_0.append(temp_pre_s.pop(test_th)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(temp_inter_s.pop(test_th)[:, np.newaxis, :, :].astype(np.float32))

            elif dataset == 'Kaggle2014Pred':
                # temp_th = deepcopy(test_th)
                # if pat in ['Dog_3', 'Dog_4']:
                #     test_th += 4

                samples_0.append(temp_pre_s.pop(test_th)[:, np.newaxis, :, :].astype(np.float32))
                samples_1.append(temp_inter_s.pop(test_th)[:, np.newaxis, :, :].astype(np.float32))

                # test_th = temp_th
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

        if method in ['original', 'random labels', 'kd-reciprocal']:  # 使用了unlearning patients的数据
            in_out_labels_0.append(np.ones(len(samples_0[index]), dtype=np.int64))
            in_out_labels_1.append(np.ones(len(samples_1[index]), dtype=np.int64))
        elif method in ['finetune', 'retrain', ]:
            in_out_labels_0.append(np.zeros(len(samples_0[index]), dtype=np.int64))
            in_out_labels_1.append(np.zeros(len(samples_1[index]), dtype=np.int64))

    pre_samples = np.concatenate(samples_0)
    in_out_labels_0 = np.concatenate(in_out_labels_0)
    inter_samples = np.concatenate(samples_1)
    in_out_labels_1 = np.concatenate(in_out_labels_1)

    pre_samples = torch.from_numpy(pre_samples)
    in_out_labels_0 = torch.from_numpy(in_out_labels_0)
    inter_samples = torch.from_numpy(inter_samples)
    in_out_labels_1 = torch.from_numpy(in_out_labels_1)

    pre_set = data.TensorDataset(pre_samples, in_out_labels_0)
    pre_loader = data.DataLoader(
        dataset=pre_set,
        shuffle=False,
        batch_size=64,
        num_workers=0
    )

    inter_set = data.TensorDataset(inter_samples, in_out_labels_1)
    inter_loader = data.DataLoader(
        dataset=inter_set,
        shuffle=False,
        batch_size=64,
        num_workers=0
    )

    return pre_loader, inter_loader


def load_data_balance_not_mix_data(dataset, test_th, method, *patients):
    """
    get patients' data and obtain a dataloader by using the data
    :param patients: must str in list.
                    1) one variable: original patients
                    2) two variables: remaining patients and unleanring patient
    :param dataset:
    :param test_th: the index of testing seizure data
    :param method: finetune, random labels, feature remove, retrain from scratch
    :return: 1) one variable: return a dataloader used for training original model
             2) two variables: return a dataloader used for training unlearning model
    """

    with open('SETTINGS_%s.json' % dataset) as f:
        settings = json.load(f)

    remaining_patients = patients[0]
    unlearning_patient = patients[1]
    # remaining_rate = 0.1
    len_remaining = 0
    len_unlearning = 0

    re_samples_0, re_labels_0, re_samples_1, re_labels_1 = [], [], [], []
    un_samples_0, un_labels_0_train, un_samples_1, un_labels_1_train = [], [], [], []
    un_labels_0_mia, un_labels_1_mia = [], []

    unlearning_data_rate = 1.0
    # balancing unlearning data and remaining data
    if method in ['finetune']:
        remaining_rate = 1.0
    else:
        remaining_rate = 0.1

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
            re_samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
            re_samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

            # # balance the two classes
            if len(re_samples_1[index]) >= len(re_samples_0[index]):
                np.random.shuffle(re_samples_1[index])
                re_samples_1[index] = re_samples_1[index][:len(re_samples_0[index])]
            else:
                np.random.shuffle(re_samples_0[index])
                re_samples_0[index] = re_samples_0[index][:len(re_samples_1[index])]

        elif dataset == 'Kaggle2014Pred':
            temp_pre_s.pop(test_th)
            re_samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
            re_samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

        else:
            raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                             "but got dataset={}".format(dataset))

        # # get subset with remaining_rate of each remaining patient
        re_samples_0[index] = re_samples_0[index][:int(len(re_samples_0[index]) * remaining_rate)]
        re_samples_1[index] = re_samples_1[index][:int(len(re_samples_1[index]) * remaining_rate)]
        # # '0' and '1' for preictal and interictal segments of remaining patients, respectively
        re_labels_0.append(np.ones(len(re_samples_0[index]), dtype=np.int64))
        re_labels_1.append(np.zeros(len(re_samples_1[index]), dtype=np.int64))
        len_remaining += (len(re_samples_0[index]) + len(re_samples_1[index]))

    # get unlearning patients' data
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
            un_samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
            un_samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

            # # balance the two classes
            if len(un_samples_1[index]) >= len(un_samples_0[index]):
                np.random.shuffle(un_samples_1[index])
                un_samples_1[index] = un_samples_1[index][:len(un_samples_0[index])]
            else:
                np.random.shuffle(un_samples_0[index])
                un_samples_0[index] = un_samples_0[index][:len(un_samples_1[index])]

        elif dataset == 'Kaggle2014Pred':
            temp_pre_s.pop(test_th)
            un_samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
            un_samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

        else:
            raise ValueError("dataset expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                             "but got dataset={}".format(dataset))

        un_samples_0[index] = un_samples_0[index][:int(len(un_samples_0[index]) *
                                                       unlearning_data_rate)]
        un_samples_1[index] = un_samples_1[index][:int(len(un_samples_1[index]) *
                                                       unlearning_data_rate)]

        if method == 'random labels':
            # # random labels
            un_labels_0_train[index] = np.random.randint(0, 2, len(un_samples_0[index]), dtype=int)
            un_labels_1_train[index] = np.random.randint(0, 2, len(un_samples_1[index]), dtype=int)
        else:
            # # reverse labels 1
            un_labels_0_train.append(np.ones(len(un_samples_0[index]), dtype=np.int64))
            un_labels_1_train.append(np.zeros(len(un_samples_1[index]), dtype=np.int64))

        # for membership inference attack
        un_labels_0_mia.append(np.ones(len(un_samples_0[index]), dtype=np.int64))
        un_labels_1_mia.append(np.zeros(len(un_samples_1[index]), dtype=np.int64))

        len_unlearning += (len(un_samples_0[index]) + len(un_samples_1[index]))

    if method in ['finetune', 'retrain', 'kd-reciprocal', 'original']:
        # not use the remaining loader for the last two method

        patients_data_re = {'p_s': re_samples_0, 'p_l': re_labels_0, 'i_s': re_samples_1, 'i_l': re_labels_1}

        remaining_samples, remaining_labels = [], []
        for j in range(len(patients_data_re['p_s'])):
            # # append preictal
            remaining_samples.append(patients_data_re['p_s'][j])
            remaining_labels.append(patients_data_re['p_l'][j])

            # # append interictal
            remaining_samples.append(patients_data_re['i_s'][j])
            remaining_labels.append(patients_data_re['i_l'][j])

        remaining_samples = np.concatenate(remaining_samples)
        remaining_labels = np.concatenate(remaining_labels)

        remaining_samples = torch.from_numpy(remaining_samples)
        remaining_labels = torch.from_numpy(remaining_labels)

        remaining_set = data.TensorDataset(remaining_samples, remaining_labels)
        remaining_loader = data.DataLoader(
            dataset=remaining_set,
            shuffle=True,
            batch_size=64,
            num_workers=0
        )

        patients_data_un = {'p_s': un_samples_0, 'p_l': un_labels_0_mia,
                            'i_s': un_samples_1, 'i_l': un_labels_1_mia}

        attack_samples, attack_labels = [], []
        for j in range(len(patients_data_un['p_s'])):
            # # append preictal
            attack_samples.append(patients_data_un['p_s'][j])
            attack_labels.append(patients_data_un['p_l'][j])

            # # append interictal
            attack_samples.append(patients_data_un['i_s'][j])
            attack_labels.append(patients_data_un['i_l'][j])

        attack_samples = np.concatenate(attack_samples)
        attack_labels = np.concatenate(attack_labels)

        attack_samples = torch.from_numpy(attack_samples)
        attack_labels = torch.from_numpy(attack_labels)

        attack_set = data.TensorDataset(attack_samples, attack_labels)
        attack_loader = data.DataLoader(
            dataset=attack_set,
            shuffle=False,
            batch_size=64,
            num_workers=0
        )

        return remaining_loader, attack_loader

    elif method in ['random labels']:

        patients_data_un = {'p_s': un_samples_0, 'p_l': un_labels_0_mia, 'i_s': un_samples_1, 'i_l': un_labels_1_mia}

        attack_samples, attack_labels = [], []
        for j in range(len(patients_data_un['p_s'])):
            # # append preictal
            attack_samples.append(patients_data_un['p_s'][j])
            attack_labels.append(patients_data_un['p_l'][j])

            # # append interictal
            attack_samples.append(patients_data_un['i_s'][j])
            attack_labels.append(patients_data_un['i_l'][j])

        attack_samples = np.concatenate(attack_samples)
        attack_labels = np.concatenate(attack_labels)

        attack_samples = torch.from_numpy(attack_samples)
        attack_labels = torch.from_numpy(attack_labels)

        attack_set = data.TensorDataset(attack_samples, attack_labels)
        attack_loader = data.DataLoader(
            dataset=attack_set,
            shuffle=False,
            batch_size=64,
            num_workers=0
        )

        re_samples_0.append(un_samples_0[0])
        re_labels_0.append(un_labels_0_train[0])
        re_samples_1.append(un_samples_1[0])
        re_labels_1.append(un_labels_1_train[0])
        patients_data_un_re = {'p_s': re_samples_0, 'p_l': re_labels_0, 'i_s': re_samples_1, 'i_l': re_labels_1}

        remaining_samples, remaining_labels = [], []
        for j in range(len(patients_data_un_re['p_s'])):
            # # append preictal
            remaining_samples.append(patients_data_un_re['p_s'][j])
            remaining_labels.append(patients_data_un_re['p_l'][j])

            # # append interictal
            remaining_samples.append(patients_data_un_re['i_s'][j])
            remaining_labels.append(patients_data_un_re['i_l'][j])

        remaining_samples = np.concatenate(remaining_samples)
        remaining_labels = np.concatenate(remaining_labels)

        remaining_samples = torch.from_numpy(remaining_samples)
        remaining_labels = torch.from_numpy(remaining_labels)

        domain_labels = np.concatenate((np.zeros(len_remaining, dtype=np.int64),
                                        np.ones(len_unlearning, dtype=np.int64)))

        domain_labels = torch.from_numpy(domain_labels)

        remaining_set = data.TensorDataset(remaining_samples, remaining_labels, domain_labels)
        remaining_loader = data.DataLoader(
            dataset=remaining_set,
            shuffle=True,
            batch_size=64,
            num_workers=0
        )

        return remaining_loader, attack_loader

    else:
        raise ValueError("method expected specified option in 'finetune', 'retrain', 'random labels', "
                         "and 'feature unlearning' but got dataset={}".format(method))


def retrain_featurer_mia(original_model, unlearning_loader, device, lr, epochs):
    """

    :param original_model:
    :param unlearning_loader:
    :param device:
    :param lr:
    :param epochs:
    :return:
    """

    # retrain classifier
    # # update classifier parameters

    # retrain classifier
    # # update classifier parameters
    unlearning_model = deepcopy(original_model)

    # test_model = deepcopy(original_model)
    # bound = 0.1
    # norm = False
    # random_start = False
    # adv = FGSM(test_model, bound, norm, random_start, device)

    optimizer = optim.Adam(unlearning_model.featurer.parameters(), lr=lr)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    criterion_ce = nn.CrossEntropyLoss()
    original_model.to(device)
    original_model.eval()
    unlearning_model.to(device)
    unlearning_model.train()
    for param in unlearning_model.classifier.parameters():
        param.required_grad = False

    for epoch in range(epochs):
        loop = tqdm(unlearning_loader, ascii=True)
        training_loss = []
        for samples, labels_class, labels_domain in loop:

            samples = samples.to(device)
            labels_class = labels_class.to(device)

            optimizer.zero_grad()
            dynamic_logits = unlearning_model(samples)
            static_logits = original_model(samples)

            remaining_logits_dynamic, unlearning_logits_dynamic, remaining_labels, unlearning_labels, \
                remaining_logits_static = separate_prediction([labels_class], labels_domain, device,
                                                              static_logits, dynamic_logits)

            loss_unlearn = criterion_ce(unlearning_logits_dynamic, unlearning_labels)

            loss_remain = criterion_kl(torch.log(func.softmax(remaining_logits_dynamic, dim=-1)),
                                       func.softmax(remaining_logits_static, dim=-1))
            loss_T = loss_Tsallis(remaining_logits_dynamic)
            alpha = 5.0
            beta = 1.0
            gamma = 1.0

            loss = alpha * loss_unlearn + beta * loss_remain + gamma * loss_T
            # loss = alpha * loss_unlearn + beta * loss_remain
            loss.backward()

            optimizer.step()

            training_loss.append(loss.item() / labels_class.shape[0])
            loop.set_description(
                f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
            )

        loop.close()

    return unlearning_model


def separate_samples(samples, labels, domain_labels, device):

    domain_labels_numpy = deepcopy(domain_labels).cpu().numpy()
    num_u = len(domain_labels_numpy[domain_labels_numpy == 1])
    num_r = len(domain_labels_numpy[domain_labels_numpy == 0])
    remaining_samples, remaining_labels, unlearning_samples, unlearning_labels = [], [], [], []

    for ind, label in enumerate(domain_labels_numpy):
        if int(label) == 0:
            remaining_samples.append(samples[ind])
            remaining_labels.append(labels[ind])
        elif int(label) == 1:
            unlearning_samples.append(samples[ind])
            unlearning_labels.append(labels[ind])
        else:
            print('unexcepted labels')
            exit()

    try:
        remaining_samples = torch.cat(remaining_samples).view(num_r, -1)
        remaining_labels = torch.tensor(remaining_labels, dtype=torch.int64, device=device)
    except NotImplementedError:
        unlearning_samples = torch.zeros((1, 2), dtype=torch.float32, device=device)
        remaining_labels = torch.zeros((1,), dtype=torch.int64, device=device)
        print('lack remaining samples')
        pass

    try:
        unlearning_samples = torch.cat(unlearning_samples).view(num_u, -1)
        unlearning_labels = torch.tensor(unlearning_labels, dtype=torch.int64, device=device)
    except NotImplementedError:
        unlearning_samples = torch.zeros((1, 2), dtype=torch.float32, device=device)
        unlearning_labels = torch.zeros((1,), dtype=torch.int64, device=device)
        print('lack unlearning samples')
        pass

    return remaining_samples, remaining_labels, unlearning_samples, unlearning_labels


def test_method(model_name, model, method, test_th, path, dataset, remaining_patients, unlearning_patients,
                all_patients, device):

    # finetune
    if method == 'finetune':
        remaining_loader, attack_loader = load_data_balance_not_mix_data(dataset, test_th, method,
                                                                         remaining_patients, unlearning_patients)

        # finetune
        # code for none finetune model
        if not os.path.exists(os.path.join(path['unlearn_path'],
                                           f'finetune_model_para_{unlearning_patients[0]}_{test_th}.pth')):

            unlearning_model = deepcopy(model)
            unlearning_model.load_state_dict(torch.load(os.path.join(path['original_path'],
                                                                     f'test_th_{test_th}_original_model_para.pth')))

            unlearning_model.to(device)
            unlearning_model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(unlearning_model.parameters(), lr=0.00001)
            print(f'finetune patient: {unlearning_patients[0]}, test_th: {test_th}')
            for epoch in range(5):
                looper = tqdm(remaining_loader)
                training_loss = []
                for samples, labels in looper:

                    samples = samples.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    finetune_logits = unlearning_model(samples)
                    loss = criterion(finetune_logits, labels)
                    loss.backward()

                    optimizer.step()

                    training_loss.append(loss.item() / labels.shape[0])
                    looper.set_description(
                        f"Epoch: {epoch + 1}/{5} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
                    )

                looper.close()

            torch.save(unlearning_model.state_dict(), os.path.join(path['unlearn_path'],
                                                                   f'unlearned_model_para_'
                                                                   f'{unlearning_patients[0]}_{test_th}.pth'))

        else:
            print('load exsiting original models\n')
            unlearning_model = deepcopy(model)
            unlearning_model.load_state_dict(torch.load(os.path.join(path['unlearn_path'],
                                                                     f'unlearned_model_para_'
                                                                     f'{unlearning_patients[0]}_{test_th}.pth')))

        # MIA
        test_model = deepcopy(unlearning_model)
        bound = 0.1
        norm = False
        random_start = False
        adv = FGSM(test_model, bound, norm, random_start, device)

        test_model.to(device)
        test_model.eval()

        print('memebership inference attack')
        for epoch in range(3):
            looper = tqdm(attack_loader)
            num_hits = 0
            num_sum = 0
            for samples, labels in looper:

                samples = samples.to(device)
                labels = labels.to(device)

                # MIA
                x_adv = adv.perturb(samples, labels, target_y=None, model=test_model, device=device)
                adv_logits = test_model(x_adv)
                pred_label = torch.argmax(adv_logits, dim=1)
                num_hits += (labels != pred_label).sum()
                num_sum += labels.shape[0]

            print(f'attack patient: {unlearning_patients[0]} test_th: {test_th}')
            print(f'attack success ratio: {(num_hits / num_sum):.3f}')

        print('finetune')

    # random labels
    elif method == 'random labels':
        remaining_loader, attack_loader = load_data_balance_not_mix_data(dataset, test_th, method,
                                                                         remaining_patients, unlearning_patients)

        # finetune
        # code for none finetune model
        if not os.path.exists(os.path.join(path['unlearn_path'],
                                           f'finetune_model_para_{unlearning_patients[0]}_{test_th}.pth')):

            unlearning_model = deepcopy(model)
            # unlearning_model.load_state_dict(torch.load(os.path.join(path['original_path'],
            #                                                          f'test_th_{test_th}_original_model_para.pth')))

            unlearning_model.to(device)
            unlearning_model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(unlearning_model.parameters(), lr=0.0001)
            print(f'finetune patient: {unlearning_patients[0]}, test_th: {test_th}')
            for epoch in range(5):
                looper = tqdm(remaining_loader)
                training_loss = []
                for samples, labels, domain_labels in looper:

                    # remaining_samples, remaining_labels, unlearning_samples, unlearning_labels = \
                    #     separate_samples(samples, labels, domain_labels, device)

                    samples = samples.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    logits = unlearning_model(samples)
                    loss = criterion(logits, labels)
                    loss.backward()

                    optimizer.step()

                    training_loss.append(loss.item() / labels.shape[0])
                    looper.set_description(
                        f"Epoch: {epoch + 1}/{5} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
                    )

                looper.close()

            torch.save(unlearning_model.state_dict(), os.path.join(path['unlearn_path'],
                                                                   f'unlearned_model_para_'
                                                                   f'{unlearning_patients[0]}_{test_th}.pth'))

        else:
            print('load exsiting original models\n')
            unlearning_model = deepcopy(model)
            unlearning_model.load_state_dict(torch.load(os.path.join(path['unlearn_path'],
                                                                     f'unlearned_model_para_'
                                                                     f'{unlearning_patients[0]}_{test_th}.pth')))

        # MIA
        test_model = deepcopy(unlearning_model)
        bound = 0.1
        norm = False
        random_start = False
        adv = FGSM(test_model, bound, norm, random_start, device)

        test_model.to(device)
        test_model.eval()

        print('memebership inference attack')
        for epoch in range(3):
            looper = tqdm(attack_loader)
            num_hits = 0
            num_sum = 0
            for samples, labels in looper:
                samples = samples.to(device)
                labels = labels.to(device)

                # MIA
                x_adv = adv.perturb(samples, labels, target_y=None, model=test_model, device=device)
                adv_logits = test_model(x_adv)
                pred_label = torch.argmax(adv_logits, dim=1)
                num_hits += (labels != pred_label).sum()
                num_sum += labels.shape[0]

            print(f'attack patient: {unlearning_patients[0]} test_th: {test_th}')
            print(f'attack success ratio: {(num_hits / num_sum):.3f}')

        print('random labels')

    # feature unlearning (our method)
    elif method == 'kd-reciprocal':

        name_ = ''
        for ind_, pat_ in enumerate(unlearning_patient_):
            if ind_ < len(unlearning_patient_) - 1:
                name_ += pat_ + '+'
            else:
                name_ += pat_

        unlearning_model = deepcopy(model)
        unlearning_model.load_state_dict(torch.load(os.path.join(path['unlearn_path'],
                                                                 f'test_th_{test_th}_{name_}'
                                                                 f'_unlearned_model_para.pth')))
        # MIA
        # # obtain attack model
        attack_model_path = f'models/{dataset}/attack/{model_name}'
        os.makedirs(attack_model_path, exist_ok=True)
        num_classes = 2

        if not os.path.exists(os.path.join(attack_model_path,
                                           f'test_th_{test_th}_category_{1}_attack_model.pth')):
            raw_model = deepcopy(model)
            attack_models = obtain_attack_models(dataset, test_th, all_patients, raw_model,
                                                 device, attack_model_path)

        else:
            attack_models = []
            for i in range(num_classes):
                attack_models.append(torch.load(
                    os.path.join(attack_model_path, f'test_th_{test_th}_category_{i + 1}_attack_model.pth')))

        membership_inference_attack(dataset, test_th, unlearning_patients, method,
                                    unlearning_model, attack_models, device)

        print('kd reciprocal')

    # retrain from scratch
    elif method == 'retrain':

        # finetune
        # code for none finetune model
        name_ = ''
        for ind_, pat_ in enumerate(unlearning_patient_):
            if ind_ < len(unlearning_patient_) - 1:
                name_ += pat_ + '+'
            else:
                name_ += pat_

        if not os.path.exists(os.path.join(path['retrain_path'],
                                           f'except_{name_}_para.pth')):

            remaining_loader, attack_loader = load_data_balance_not_mix_data(dataset, test_th, method,
                                                                             remaining_patients, unlearning_patients)

            retraining_model = deepcopy(model)

            retraining_model.to(device)
            retraining_model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(retraining_model.parameters(), lr=0.0001)
            print(f'retrain without patient: {unlearning_patients[0]}, test_th: {test_th}')
            for epoch in range(5):
                looper = tqdm(remaining_loader)
                training_loss = []
                for samples, labels in looper:

                    samples = samples.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    logits = retraining_model(samples)
                    loss = criterion(logits, labels)
                    loss.backward()

                    optimizer.step()

                    training_loss.append(loss.item() / labels.shape[0])
                    looper.set_description(
                        f"Epoch: {epoch + 1}/{5} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
                    )

                looper.close()

            torch.save(retraining_model.state_dict(), os.path.join(path['retrain_path'],
                                                                   f'except_{name_}_para.pth'))

        else:
            print('load exsiting retrained models\n')
            retraining_model = deepcopy(model)
            retraining_model.load_state_dict(torch.load(os.path.join(path['retrain_path'],
                                                                     f'except_{name_}_para.pth')))

        # MIA
        # # obtain attack model
        attack_model_path = f'models/{dataset}/attack/{model_name}'
        os.makedirs(attack_model_path, exist_ok=True)
        num_classes = 2

        if not os.path.exists(os.path.join(attack_model_path,
                                           f'test_th_{test_th}_category_{1}_attack_model.pth')):
            raw_model = deepcopy(model)
            attack_models = obtain_attack_models(dataset, test_th, all_patients, raw_model,
                                                 device, attack_model_path)

        else:
            attack_models = []
            for i in range(num_classes):
                attack_models.append(torch.load(
                    os.path.join(attack_model_path, f'test_th_{test_th}_category_{i + 1}_attack_model.pth')))

        membership_inference_attack(dataset, test_th, unlearning_patients, method,
                                    retraining_model, attack_models, device)

        print('retrain')

    #
    elif method == 'original':

        name_ = ''
        for ind_, pat_ in enumerate(unlearning_patient_):
            if ind_ < len(unlearning_patient_) - 1:
                name_ += pat_ + '+'
            else:
                name_ += pat_

        original_model = deepcopy(model)
        original_model.load_state_dict(torch.load(os.path.join(path['original_path'],
                                                               f'test_th_{test_th}_original_model_para.pth')))
        # MIA
        # # obtain attack model
        attack_model_path = f'models/{dataset}/attack/{model_name}'
        os.makedirs(attack_model_path, exist_ok=True)
        num_classes = 2

        if not os.path.exists(os.path.join(attack_model_path,
                                           f'test_th_{test_th}_category_{1}_attack_model.pth')):
            raw_model = deepcopy(model)
            attack_models = obtain_attack_models(dataset, test_th, all_patients, raw_model,
                                                 device, attack_model_path)

        else:
            attack_models = []
            for i in range(num_classes):
                attack_models.append(torch.load(
                    os.path.join(attack_model_path, f'test_th_{test_th}_category_{i + 1}_attack_model.pth')))

        membership_inference_attack(dataset, test_th, unlearning_patients, method,
                                    original_model, attack_models, device)

    else:
        raise ValueError("method expected specified option in 'finetune', 'random labels', 'feature unlearning', "
                         "and 'retrain' but got aug_pertur={}".format(method))


if __name__ == '__main__':

    dataset_ = ['CHBMIT', 'Kaggle2014Pred']
    used_dataset_ = dataset_[0]

    models = {'CHBMIT': {'ViT': ViT(), 'PCT': PCT(), 'RepNet': RepNet(22),
                         'CNN': CNN()},
              'Kaggle2014Pred': {'ViT': ViT_kaggle(), 'RepNet': RepNet(16), 'CNN': CNN_kaggle()}}

    model_name_ = 'MLP'
    used_model = models[used_dataset_][model_name_]

    methods_ = ['SISA']
    used_method_ = methods_[0]

    para_pathes = {'original_path': f'models/{used_dataset_}/only_unlearning/{model_name_}/basic_model_mia',
                   'unlearn_path': f'models/{used_dataset_}/only_unlearning/{model_name_}/unlearned_model_mia',
                   'retrain_path': f'models/{used_dataset_}/retrain/{model_name_}'}

    # results_path_ = {'original': f'results/MIA/original/{used_dataset_}/{model_name}',
    #                  'unlearn': f'results/MIA/unlearn/{used_method_}/{used_dataset_}/{model_name}'}
    # os.makedirs(results_path_['original'], exist_ok=True)
    # os.makedirs(results_path_['unlearn'], exist_ok=True)

    device_ = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    all_patients_, skip_patients = [], []
    if used_dataset_ == 'CHBMIT':

        all_patients_ = [  # more than 3 seizure data
            '1',  #
            # '2',  #
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
            # '22',  #
            '23'  #
        ]
        skip_patients = [
            # '1',  #
            # '3',  #
            # '5',  #
            # '6',  #
            # '8',  #
            # '10',  #
            # '13',  #
            # '14',  #
            # '16',  #
            # '18',  #
            # '20',  #
            # '23'  #
        ]

    elif used_dataset_ == 'Kaggle2014Pred':

        all_patients_ = [
            # 'Dog_1',
            'Dog_2',
            'Dog_3',
            'Dog_4',
            'Dog_5',
            # 'Patient_1',
            # 'Patient_2',
        ]
        skip_patients = [
            # 'Dog_1',
            # 'Dog_2',
            # 'Dog_3',
            # 'Dog_4',
            # 'Dog_5',
            # 'Patient_1',
            # 'Patient_2',
        ]

    else:

        print('unexcepted dataset')
        exit()

    # settings for training orginal models
    lr_o = 0.0001
    epoch_o = 10

    # test_th_list = [-3, -2, -1]  # used in CHBMIT
    # test_th_list = [1, 2, 3]
    # test_th_list = [0, 1, 2]
    test_th_list = [-5, -4, -3, -2, -1]
    for th in test_th_list:

        if th in [
            # -3,
            # -2,
            # -1
        ]:
            print(f'skip test_th : {th}')
            continue

        else:
            print(f'test_th = {th}')

        # settings for training unlearning models
        # lr_u = [0.000001, 0.000001]  # used for CNN model temporally
        lr_u = [0.0001, 0.0001]
        epoch_u = 2
        for patient_ in all_patients_:
            # # skip some patients (patients whose data have already unlearned)
            if patient_ in skip_patients:
                print(f'skip case: {patient_}\n')
                continue

            unlearning_patient_ = [patient_]
            remaining_patients_ = deepcopy(all_patients_)
            remaining_patients_.remove(patient_)

            # unlearn_methods_ = ['shadow states', 'influence functions', 'retrain calssifier']
            # disturb_methods_ = ['samples', 'labels', 'samples_labels']

            #
            test_method(model_name_, used_model, used_method_, th, para_pathes, used_dataset_,
                        remaining_patients_, unlearning_patient_, all_patients_, device_)

    print('main')
