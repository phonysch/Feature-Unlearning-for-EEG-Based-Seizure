import os
import csv
from copy import deepcopy
import torch
import json
import numpy as np
from load_data import PrepData
import torch.utils.data as data
from PCT_net import PCT
from PCT_net_kaggle import PCT as PCT_kaggle
from CNN import CNN_kaggle
from CNN import CNN
from tqdm import tqdm
from STMLP import STMLP
from Transformer import ViT
from Transformer import ViT_kaggle
from STMLP_configs import configs
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
from sklearn import metrics
import random


def get_data(database_name, patients):

    print('get_data()')

    with open('SETTINGS_%s.json' % database_name) as f:
        settings = json.load(f)

    #
    samples_0, labels_0, samples_1, labels_1 = [], [], [], []
    train_samples, train_labels, test_samples, test_labels = [], [], [], []
    test_rate = 0.1
    for index, pat in enumerate(patients):  # '10' -> 1, 0
        temp_pre_s, temp_pre_l = PrepData(pat, type='ictal', settings=settings).apply()
        temp_inter_s, temp_inter_l = PrepData(pat, type='interictal', settings=settings).apply()

        if database_name == 'CHBMIT':
            samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
            samples_1.append(np.concatenate(temp_inter_s)[:, np.newaxis, :, :].astype(np.float32))

        elif database_name == 'Kaggle2014Pred':
            samples_0.append(np.concatenate(temp_pre_s)[:, np.newaxis, :, :].astype(np.float32))
            samples_1.append(temp_inter_s[:, np.newaxis, :, :].astype(np.float32))

        else:
            raise ValueError("database_name expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                             "but got database_name={}".format(database_name))

        # # balance the two classes
        if len(samples_1[index]) >= len(samples_0[index]):
            np.random.shuffle(samples_1[index])
            samples_1[index] = samples_1[index][:len(samples_0[index])]
        else:
            np.random.shuffle(samples_0[index])
            samples_0[index] = samples_0[index][:len(samples_1[index])]

        train_samples.append(np.concatenate((samples_0[index][0:-int(len(samples_0[index])*test_rate)],
                                             samples_1[index][0:-int(len(samples_0[index])*test_rate)])))

        test_samples.append(np.concatenate((samples_0[index][-int(len(samples_0[index])*test_rate):],
                                            samples_1[index][-int(len(samples_0[index])*test_rate):])))

        train_labels.append(np.concatenate((np.ones(len(samples_0[index][0:-int(len(samples_0[index])*test_rate)]),
                                                    dtype=np.int64),
                                            np.zeros(len(samples_1[index][0:-int(len(samples_0[index])*test_rate)]),
                                                     dtype=np.int64))))
        test_labels.append(np.concatenate((np.ones(len(samples_0[index][-int(len(samples_0[index])*test_rate):]),
                                                   dtype=np.int64),
                                           np.zeros(len(samples_1[index][-int(len(samples_0[index])*test_rate):]),
                                                    dtype=np.int64))))

        assert len(train_samples[index]) == len(train_labels[index])
        assert len(test_samples[index]) == len(test_labels[index])

    train_data = [train_samples, train_labels]
    test_data = [test_samples, test_labels]

    return train_data, test_data


def set_to_loader(samples, labels, train_or_test):
    """

    :param samples:
    :param labels:
    :param train_or_test: True for train set, False for test set.
    :return: train or test loader
    """

    # print('set_to_loader()')

    if train_or_test:

        samples, labels = np.concatenate(samples), np.concatenate(labels)

        samples = torch.from_numpy(samples)
        labels = torch.from_numpy(labels)

        train_set = data.TensorDataset(samples, labels)
        train_loader = data.DataLoader(
            dataset=train_set,
            shuffle=True,
            batch_size=256,
            num_workers=0
        )

        return train_loader

    else:

        samples = torch.from_numpy(samples)
        labels = torch.from_numpy(labels)

        test_set = data.TensorDataset(samples, labels)
        test_loader = data.DataLoader(
            dataset=test_set,
            shuffle=False,
            batch_size=256,
            num_workers=0
        )

        return test_loader


def train_general_model_and_test(train_data, test_data, model_name, model):

    print('train_general_model_and_test()')

    para_path = f'models/general/{model_name}'
    os.makedirs(para_path, exist_ok=True)

    if not os.path.exists(os.path.join(para_path, 'general_model_para.pth')):

        print('train general model')

        # train set to train loader
        train_samples, train_labels = train_data
        train_loader = set_to_loader(train_samples, train_labels, True)

        # train general model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        general_model = deepcopy(model)
        optimizer = optim.Adam(general_model.parameters(), lr=0.001)
        criterion_ce = nn.CrossEntropyLoss()
        general_model.to(device)
        general_model.train()

        for epoch in range(10):
            loop = tqdm(train_loader, ascii=True)
            training_loss = []
            for samples, labels in loop:
                samples = samples.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = general_model(samples)
                loss = criterion_ce(logits, labels)
                loss.backward()
                optimizer.step()

                training_loss.append(loss.item() / labels.shape[0])
                loop.set_description(
                    f"Epoch: {epoch + 1}/{10} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
                )

            loop.close()

        # test on each patient's seizure data (segmetn-based)
        accuracy = []
        general_model.eval()
        with torch.no_grad():
            for ind in range(len(test_data[0])):
                test_samples, test_labels = test_data[0][ind], test_data[1][ind]
                test_loader = set_to_loader(test_samples, test_labels, False)

                labels_, preds_ = [], []
                for samples, labels in test_loader:
                    samples = samples.to(device)
                    labels = labels.to(device)
                    labels_.append(labels)

                    logits = general_model(samples)
                    _, preds = torch.max(logits, dim=1)
                    preds_.append(preds)

                labels_, preds_ = torch.cat(labels_), torch.cat(preds_)
                labels_, preds_ = labels_.cpu().numpy(), preds_.cpu().numpy()
                results = np.equal(labels_, preds_)
                acc = sum(results) / len(results)
                accuracy.append(acc)

        # record testing results in '.csv' file
        save_path = 'results/general/'
        os.makedirs(save_path, exist_ok=True)
        if not os.path.exists(os.path.join(save_path, 'general_model_results.csv')):
            with open(os.path.join(save_path, 'general_model_results.csv'), 'a+', encoding='utf8', newline='') as file:
                writer = csv.writer(file)
                for a in accuracy:
                    content = [a]
                    writer.writerow(content)

        torch.save(general_model.state_dict(), os.path.join(para_path, 'general_model_para.pth'))

        return general_model

    else:
        print('load existing general model')

        general_model = deepcopy(model)
        general_model.load_state_dict(torch.load(os.path.join(para_path, 'general_model_para.pth')))

        return general_model


def data_to_loader(remaining_data, unlearning_data, remaining_rate, unlearning_rate):

    print('data_to_loader()')
    samples, labels, domain_labels = [], [], []

    # append partial remaining data
    for i in range(len(remaining_data[0])):
        sample, label = remaining_data[0][i], remaining_data[1][i]
        shuffle_index = np.random.permutation(len(sample))
        samples.append(sample[shuffle_index][0:int(len(sample) * remaining_rate)])
        labels.append(label[shuffle_index][0:int(len(sample) * remaining_rate)])
        domain_labels.append(np.zeros(len(labels[i])))  # '0' for remaining samples, '1' for unlearning samples

    # append partial unlearning data
    sample, label = unlearning_data[0][0], unlearning_data[1][0]
    shuffle_index = np.random.permutation(len(sample))
    samples.append(sample[shuffle_index][0:int(len(sample) * unlearning_rate)])
    labels.append(label[shuffle_index][0:int(len(sample) * unlearning_rate)])
    domain_labels.append(np.ones(len(labels[-1])))

    # # reverse unlearning labels
    labels[-1][labels[-1] == 0] = 2
    labels[-1][labels[-1] == 1] = 0
    labels[-1][labels[-1] == 2] = 1

    # data to loader
    samples, labels, domain_labels = np.concatenate(samples), np.concatenate(labels), np.concatenate(domain_labels)
    samples, labels, domain_labels = \
        torch.from_numpy(samples), torch.from_numpy(labels), torch.from_numpy(domain_labels)
    unlearning_set = data.TensorDataset(samples, labels, domain_labels)
    unlearning_loader = data.DataLoader(
        dataset=unlearning_set,
        shuffle=True,
        batch_size=128,
        num_workers=0
    )

    return unlearning_loader


def data_to_loader_2(unlearning_data, unlearning_rate):

    print('data_to_loader_2()')
    samples, labels = [], []

    # append partial unlearning data
    sample, label = unlearning_data[0][0], unlearning_data[1][0]
    shuffle_index = np.random.permutation(len(sample))
    samples.append(sample[shuffle_index][0:int(len(sample) * unlearning_rate)])
    labels.append(label[shuffle_index][0:int(len(sample) * unlearning_rate)])

    # # reverse unlearning labels
    labels[-1][labels[-1] == 0] = 2
    labels[-1][labels[-1] == 1] = 0
    labels[-1][labels[-1] == 2] = 1

    # data to loader
    samples, labels = np.concatenate(samples), np.concatenate(labels)
    samples, labels = torch.from_numpy(samples), torch.from_numpy(labels)
    forgetting_set = data.TensorDataset(samples, labels)
    forgetting_loader = data.DataLoader(
        dataset=forgetting_set,
        shuffle=True,
        batch_size=128,
        num_workers=0
    )

    return forgetting_loader


def separate_predictions(labels_class, labels_domain, *predictions):
    """

    :param labels_class:
    :param labels_domain:
    :param predictions:
    :return:
    """

    # print('seprerate_predictions()')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    labels_domain_numpy = deepcopy(labels_domain).cpu().numpy()
    num_u = len(labels_domain_numpy[labels_domain_numpy == 1])
    num_r = len(labels_domain_numpy[labels_domain_numpy == 0])
    unlearning_labels, remaining_labels = [], []

    remaining_logits_static, remaining_logits_dynamic, unlearning_logits_dynamic, = [], [], []
    for ind, label in enumerate(labels_domain_numpy):
        if int(label) == 0:
            remaining_logits_static.append(predictions[0][ind])
            remaining_logits_dynamic.append(predictions[1][ind])
            remaining_labels.append(labels_class[ind])
        elif int(label) == 1:
            unlearning_logits_dynamic.append(predictions[1][ind])
            unlearning_labels.append(labels_class[ind])

        else:
            print('unexcepted labels')
            exit()

    try:
        remaining_logits_static = torch.cat(remaining_logits_static).view(num_r, -1)
        remaining_logits_dynamic = torch.cat(remaining_logits_dynamic).view(num_r, -1)
        remaining_labels = torch.tensor(remaining_labels, dtype=torch.int64, device=device)
    except NotImplementedError:
        remaining_logits_static = torch.zeros((1, 2), dtype=torch.float32, device=device)
        remaining_logits_dynamic = torch.zeros((1, 2), dtype=torch.float32, device=device)
        remaining_labels = torch.zeros((1,), dtype=torch.int64, device=device)
        print('lack remaining samples')
        pass

    try:
        unlearning_logits_dynamic = torch.cat(unlearning_logits_dynamic).view(num_u, -1)
        unlearning_labels = torch.tensor(unlearning_labels, dtype=torch.int64, device=device)
    except NotImplementedError:
        unlearning_logits_dynamic = torch.zeros((1, 2), dtype=torch.float32, device=device)
        unlearning_labels = torch.zeros((1,), dtype=torch.int64, device=device)
        # print('lack unlearning samples')
        pass

    return remaining_logits_dynamic, unlearning_logits_dynamic, remaining_labels, unlearning_labels, \
        remaining_logits_static


def augmentation(samples, labels_class, labels_domain):

    # print('augmentation()')

    labels_domain_numpy = deepcopy(labels_domain).cpu().numpy()

    augment_samples, augment_labels_class, augment_labels_domain, augment_complementary_labels = [], [], [], []
    for ind, label in enumerate(labels_domain_numpy):
        if int(label) == 0:
            augment_samples.append(samples[ind])
            augment_labels_class.append(labels_class[ind])
            augment_labels_domain.append(labels_domain[ind])
            # augment_complementary_labels.append(complementary_labels[ind])
        elif int(label) == 1:
            augment_samples.append(samples[ind])
            augment_labels_class.append(labels_class[ind])
            augment_labels_domain.append(labels_domain[ind])
            # augment_complementary_labels.append(complementary_labels[ind])
        else:
            print('unexcepted labels')
            exit()

    augment_samples = torch.cat(augment_samples)
    augment_samples = augment_samples.numpy()
    for sample in augment_samples:
        channel_changing_index = random.sample(range(0, sample.shape[1]), int(sample.shape[1] / 2))
        factor = np.random.rand(int(sample.shape[1] / 2), )
        for i in range(int(sample.shape[1] / 2)):
            sample[:, channel_changing_index[i]] = sample[:, channel_changing_index[i]] * factor[i]
    augment_samples = augment_samples[:, np.newaxis, :, :]
    augment_samples = torch.from_numpy(augment_samples)

    augment_labels_class = torch.tensor(augment_labels_class)
    augment_labels_domain = torch.tensor(augment_labels_domain)
    # augment_complementary_labels = torch.tensor(augment_complementary_labels)

    mixed_samples = torch.cat((samples, augment_samples))
    mixed_labels_class = torch.cat((labels_class, augment_labels_class))
    mixed_labels_domain = torch.cat((labels_domain, augment_labels_domain))
    # mixed_complementary_labels = torch.cat((complementary_labels, augment_complementary_labels))

    return mixed_samples, mixed_labels_class, mixed_labels_domain
    # return mixed_samples, mixed_labels_class, mixed_labels_domain, mixed_complementary_labels


def reverse_label(remaining_data, unlearning_data, general_model):

    print('reverse_label()')
    remaining_rate = 0.1
    unlearning_rate = 0.5
    unlearning_loader = data_to_loader(remaining_data, unlearning_data, remaining_rate, unlearning_rate)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    unlearning_model = deepcopy(general_model)
    optimizer = optim.Adam(unlearning_model.featurer.parameters(), lr=0.0001)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    general_model.to(device)
    general_model.eval()
    unlearning_model.to(device)
    unlearning_model.train()
    for param in unlearning_model.classifier.parameters():
        param.required_grad = False

    unlearning_epoch = 5
    for epoch in range(unlearning_epoch):
        loop = tqdm(unlearning_loader, ascii=True)
        training_loss = []
        for samples, labels, domain_labels in loop:

            weaken = False
            if weaken:  # want to increase unlearning effects, when unleanring on a few data.
                samples, labels, domain_labels = augmentation(samples, labels, domain_labels)

            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            dynamic_logits = unlearning_model(samples)
            static_logits = general_model(samples)

            remaining_logits_dynamic, unlearning_logits_dynamic, remaining_labels, unlearning_labels, \
                remaining_logits_static = separate_predictions(labels, domain_labels, static_logits, dynamic_logits)

            loss_unlearn = criterion_ce(unlearning_logits_dynamic, unlearning_labels)

            loss_remain = criterion_kl(torch.log(func.softmax(remaining_logits_dynamic, dim=-1)),
                                       func.softmax(remaining_logits_static, dim=-1))
            alpha = 1.0
            beta = 1.0

            loss = alpha * loss_unlearn + beta * loss_remain
            loss.backward()

            optimizer.step()

            training_loss.append(loss.item() / labels.shape[0])
            loop.set_description(
                f"Epoch: {epoch + 1}/{unlearning_epoch} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
            )

        loop.close()

    return unlearning_model


def feature_level_unlearning(remaining_data, unlearning_data, general_model, raw_model):

    print('feature level unlearing')

    # train a unlearning teacher model on remaining data and unlearing data (reverse labels)
    remaining_rate_1 = 0.1
    forgetting_rate = 0.5
    forgetting_loader = data_to_loader(remaining_data, unlearning_data, remaining_rate_1, forgetting_rate)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    unlearning_teacher_model = deepcopy(raw_model)
    optimizer = optim.Adam(unlearning_teacher_model.parameters(), lr=0.0001)
    criterion_ce = nn.CrossEntropyLoss()
    unlearning_teacher_model.to(device)
    unlearning_teacher_model.train()

    print('train unlearning teacher model')
    forgetting_epoch = 5
    for epoch in range(forgetting_epoch):
        loop = tqdm(forgetting_loader, ascii=True)
        training_loss = []
        for samples, labels, domain_labels in loop:

            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = unlearning_teacher_model(samples)
            loss = criterion_ce(logits, labels)
            loss.backward()
            optimizer.step()

            training_loss.append(loss.item() / labels.shape[0])
            loop.set_description(
                f"Epoch: {epoch + 1}/{forgetting_epoch} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
            )

        loop.close()

    # unlearning features under general and unlearning teacher model
    remaining_rate = 0.1
    unlearning_rate = 0.5
    unlearning_loader = data_to_loader(remaining_data, unlearning_data, remaining_rate, unlearning_rate)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    unlearning_model = deepcopy(general_model)
    optimizer = optim.Adam(unlearning_model.featurer.parameters(), lr=0.0001)
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    criterion_mse = nn.MSELoss()
    general_model.to(device)
    general_model.eval()
    unlearning_model.to(device)
    unlearning_model.train()
    for param in unlearning_model.classifier.parameters():
        param.required_grad = False

    print('train unlearing model')
    unlearning_epoch = 5
    for epoch in range(unlearning_epoch):
        loop = tqdm(unlearning_loader, ascii=True)
        training_loss = []
        for samples, labels, domain_labels in loop:

            weaken = False
            if weaken:  # want to increase unlearning effects, when unleanring on a few data.
                samples, labels, domain_labels = augmentation(samples, labels, domain_labels)

            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            dynamic_logits = unlearning_model(samples)
            dynamic_features = unlearning_model.featurer(samples)
            static_logits = general_model(samples)
            static_features = unlearning_teacher_model.featurer(samples)

            remaining_logits_dynamic, unlearning_logits_dynamic, remaining_labels, unlearning_labels, \
                remaining_logits_static = separate_predictions(labels, domain_labels, static_logits, dynamic_logits)

            loss_unlearn = criterion_mse(dynamic_features, static_features)

            # loss_unlearn = criterion_kl(torch.log(func.softmax(dynamic_features, dim=-1)),
            #                             func.softmax(static_features, dim=-1))

            loss_remain = criterion_kl(torch.log(func.softmax(remaining_logits_dynamic, dim=-1)),
                                       func.softmax(remaining_logits_static, dim=-1))
            alpha = 1.0
            beta = 1.0

            loss = alpha * loss_unlearn + beta * loss_remain
            loss.backward()

            optimizer.step()

            training_loss.append(loss.item() / labels.shape[0])
            loop.set_description(
                f"Epoch: {epoch + 1}/{unlearning_epoch} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
            )

        loop.close()

    return unlearning_model


def prep_test(test_data):

    # print('prep_test()')

    test_loader_list = []

    for i in range(len(test_data[0])):

        samples, labels = test_data[0][i], test_data[1][i]

        samples = torch.from_numpy(samples.astype(np.float32))
        labels = torch.from_numpy(labels.astype(np.int64))

        test_set = data.TensorDataset(samples, labels)
        test_loader = data.DataLoader(
            dataset=test_set,
            batch_size=64,
            shuffle=False,
            num_workers=0,
        )

        test_loader_list.append(test_loader)

    return test_loader_list


def test_metrics(dataloader, model):
    """
    testing on adaptation data and remain data, and giving metrixs of Sn, SEN, AUC and FPR and statistical analysis of
    p-value.
    :param dataloader:
    :param model: retrained model
    :return: None
    """
    sensitivity, specificity, auc, fpr, f1_score, precision, acc = [], [], [], [], [], [], []
    corrects = 0
    num_samples = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    # below two lines could warning when running
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

    metrics_ = [sensitivity, specificity, precision, acc, f1_score, auc, fpr]

    return metrics_


def evaluate(test_data, unlearned_model, general_model, unlearning_patient_index, patients, dataset_name, model_name):

    print('evaluate()')

    unlearning_patient = patients[unlearning_patient_index]
    remaining_patients = deepcopy(patients)
    remaining_patients.remove(unlearning_patient)
    remaining_data, unlearning_data = [[], []], [[], []]
    for i in range(len(test_data[0])):
        if i == unlearning_patient_index:
            unlearning_data[0].append(test_data[0][i])
            unlearning_data[1].append(test_data[1][i])
        else:
            remaining_data[0].append(test_data[0][i])
            remaining_data[1].append(test_data[1][i])

    unlearning_results_patient = []
    mark = []

    # test on forget data
    print('\ntesting on unlearning data')
    forget_loader_list = prep_test(unlearning_data)
    print(f'patient --- {patients[unlearning_patient_index]}')
    print('\nusing original model')
    average_metrics = test_metrics(forget_loader_list[0], general_model)
    unlearning_results_patient.append(average_metrics)
    mark.append(['original', 'forget', unlearning_patient])
    print('using unlearned model')
    average_metrics = test_metrics(forget_loader_list[0], unlearned_model)
    unlearning_results_patient.append(average_metrics)
    mark.append(['unlearning', 'forget', unlearning_patient])

    # test on remain data
    print('\ntesting on remaining data')
    remain_loader_list = prep_test(remaining_data)
    for ind, pat in enumerate(remaining_patients):
        print(f'patient --- {remaining_patients[ind]}')
        print('using original model')
        average_metrics = test_metrics(remain_loader_list[ind], general_model)
        unlearning_results_patient.append(average_metrics)
        mark.append(['original', 'remain', pat])
        print('using unlearned model\n')
        average_metrics = test_metrics(remain_loader_list[ind], unlearned_model)
        unlearning_results_patient.append(average_metrics)
        mark.append(['unlearning', 'remain', pat])

    head_mark = 0
    results_path = f'results/unlearn/retrain featurer/reserve labels/{dataset_name}/{model_name}'
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, 'results.csv'), 'a+', encoding='utf8', newline='') as file:
        writer = csv.writer(file)
        if head_mark == 0:
            writer.writerow([' ', ' ', ' ', 'Sn', 'Sp', 'precision', 'acc', 'f1_score', 'auc', 'fpr',
                             ' ', ' ', ' ', 'Sn', 'Sp', 'precision', 'acc', 'f1_score', 'auc', 'fpr'])
            head_mark += 1
        for i in range(0, len(unlearning_results_patient), 2):
            content = [mark[i][0], mark[i][1], mark[i][2],
                       sum(unlearning_results_patient[i][0]) / len(unlearning_results_patient[i][0]),
                       sum(unlearning_results_patient[i][1]) / len(unlearning_results_patient[i][1]),
                       sum(unlearning_results_patient[i][2]) / len(unlearning_results_patient[i][2]),
                       sum(unlearning_results_patient[i][3]) / len(unlearning_results_patient[i][3]),
                       sum(unlearning_results_patient[i][4]) / len(unlearning_results_patient[i][4]),
                       sum(unlearning_results_patient[i][5]) / len(unlearning_results_patient[i][5]),
                       sum(unlearning_results_patient[i][6]) / len(unlearning_results_patient[i][6]),
                       mark[i + 1][0], mark[i + 1][1], mark[i + 1][2],
                       sum(unlearning_results_patient[i + 1][0]) / len(unlearning_results_patient[i + 1][0]),
                       sum(unlearning_results_patient[i + 1][1]) / len(unlearning_results_patient[i + 1][1]),
                       sum(unlearning_results_patient[i + 1][2]) / len(unlearning_results_patient[i + 1][2]),
                       sum(unlearning_results_patient[i + 1][3]) / len(unlearning_results_patient[i + 1][3]),
                       sum(unlearning_results_patient[i + 1][4]) / len(unlearning_results_patient[i + 1][4]),
                       sum(unlearning_results_patient[i + 1][5]) / len(unlearning_results_patient[i + 1][5]),
                       sum(unlearning_results_patient[i + 1][6]) / len(unlearning_results_patient[i + 1][6])]
            writer.writerow(content)
            if i == 0:
                writer.writerow(' ')
            elif i == 1:
                writer.writerow(' ')
            else:
                pass
        writer.writerow(' ')
        writer.writerow(' ')
        writer.writerow(' ')


if __name__ == '__main__':

    dataset_ = ['CHBMIT', 'Kaggle2014Pred']
    used_dataset_ = dataset_[0]

    all_patients_, skip_patients_ = [], []
    if used_dataset_ == 'CHBMIT':

        all_patients_ = [
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
        skip_patients_ = [
            # '1',
            # '2',
            # '3',
            # '5',
            # '6',
            # '8',
            # '9',
            # '10',
            # '13',
            # '14',
            # '16',
            # '17',
            # '18',
            # '19',
            # '20',
            # '21',
            # '22',
            # '23'
        ]

    elif used_dataset_ == 'Kaggle2014Pred':

        all_patients_ = [
            'Dog_1',
            'Dog_2',
            'Dog_3',
            'Dog_4',
            'Dog_5',
            # 'Patient_1',
            # 'Patient_2',
        ]
        skip_patients_ = [
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

    models_ = {'CHBMIT': {'MLP': STMLP(**configs["ST-MLP"]), 'ViT': ViT(), 'PCT': PCT(), 'CNN': CNN()},
               'Kaggle2014Pred': {'MLP': STMLP(**configs["ST-MLP"]), 'ViT': ViT_kaggle(), 'PCT': PCT_kaggle(),
                                  'CNN': CNN_kaggle()}}

    model_name_ = 'CNN'
    used_model_ = models_[used_dataset_][model_name_]

    # load all patients' seizure data, divide the data into training and testing sets
    train_data_, test_data_ = get_data(used_dataset_, all_patients_)

    # train general model by using the training set, and test on the testing set.
    general_model_ = train_general_model_and_test(train_data_, test_data_, model_name_,  used_model_)

    # unlearn each patent's seizure data
    for ind_, patient_ in enumerate(all_patients_):
        # # skip some patients (patients whose data have already unlearned)
        if patient_ in skip_patients_:
            print(f'skip case: {patient_}\n')
            continue

        method_ = ['reverse', 'supervise']
        used_method_ = method_[1]

        print(f'train unlearning models, unlearning case: {patient_}\n')
        remaining_data_, unlearning_data_ = [[], []], [[], []]
        for i_ in range(len(train_data_[0])):
            if i_ == ind_:
                unlearning_data_[0].append(train_data_[0][i_])
                unlearning_data_[1].append(train_data_[1][i_])
            else:
                remaining_data_[0].append(train_data_[0][i_])
                remaining_data_[1].append(train_data_[1][i_])

        # # reverse and complementary labeling
        if used_method_ == 'reverse':

            unlearned_model_ = reverse_label(remaining_data_, unlearning_data_, general_model_)

        # # unlearning model supervising
        elif used_method_ == 'supervise':

            print('supervise method')
            unlearned_model_ = feature_level_unlearning(remaining_data_, unlearning_data_, general_model_, used_model_)

        else:
            raise ValueError("database_name expected specified option in 'CHBMIT' and 'Kaggle2014Pred', "
                             "but got database_name={}".format(used_method_))

        print(f'evaluate the unlearned model of case: {patient_}')
        evaluate(test_data_, unlearned_model_, general_model_, ind_, all_patients_, used_dataset_, model_name_)

    print('main_independent.py')
