import os
import csv
from copy import deepcopy
import torch
from sklearn import metrics
import numpy as np
import torch.nn.functional as func
from PCT_net import PCT
from PCT_net_kaggle import PCT as PCT_kaggle
from CNN import CNN_kaggle
from CNN import CNN
from tqdm import tqdm
from STMLP import STMLP
from Transformer import ViT
from Transformer import ViT_kaggle
from STMLP_configs import configs
from functions import prep_test1, load_data_balance
import time


def voting(sharded_models, used_device, sample):

    weight = 1 / len(sharded_models)
    out = torch.zeros([len(sample), 2], device=used_device)
    for sharded_model in sharded_models:
        sharded_model.to(used_device)
        sharded_model.eval()
        out += weight * sharded_model(sample)

    return out


def test_metrics(dataloader_list, sharded_models, used_device):
    """
    testing on adaptation data and remain data, and giving metrixs of Sn, SEN, AUC and FPR and statistical analysis of
    p-value.
    :param dataloader_list: consist of dataloader which represents one seizure data (one preictal and one interictal)
    :param sharded_models: retrained model
    :param used_device:
    :return: None
    """

    sensitivity, specificity, auc, fpr, f1_score, precision, acc = [], [], [], [], [], [], []
    corrects = 0
    num_samples = 0
    index = 0
    for dataloader in dataloader_list:
        outs, preds, labels = [], [], []
        samples = []
        with torch.no_grad():
            for sample, label in dataloader:

                sample = sample.to(used_device)
                label = label.to(used_device)

                # voting
                out = voting(sharded_models, used_device, sample)

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


# 使用sharded_model投票产生结果
def evaluate_sharded(used_patients, used_models, used_device, used_dataset, save_path):

    unlearning_results_patient = []
    mark = []

    # test on remain data
    print('\ntesting on remaining data')
    for pat in used_patients:
        remain_loader = prep_test1(used_dataset, [pat])
        print('using original model')
        average_metrics = test_metrics(remain_loader, used_models, used_device)
        unlearning_results_patient.append(average_metrics)
        mark.append(['original', 'remain', pat])

    os.makedirs(save_path, exist_ok=True)
    head_mark = 0
    with open(os.path.join(save_path, 'results.csv'), 'a+', encoding='utf8', newline='') as file:
        writer = csv.writer(file)
        if head_mark == 0:
            writer.writerow([' ', 'Sn', 'Sp', 'precision', 'acc', 'f1_score', 'auc', 'fpr'])
            head_mark += 1
        for i in range(len(unlearning_results_patient)):
            content = [mark[i][2],
                       sum(unlearning_results_patient[i][0]) / len(unlearning_results_patient[i][0]),
                       sum(unlearning_results_patient[i][1]) / len(unlearning_results_patient[i][1]),
                       sum(unlearning_results_patient[i][2]) / len(unlearning_results_patient[i][2]),
                       sum(unlearning_results_patient[i][3]) / len(unlearning_results_patient[i][3]),
                       sum(unlearning_results_patient[i][4]) / len(unlearning_results_patient[i][4]),
                       sum(unlearning_results_patient[i][5]) / len(unlearning_results_patient[i][5]),
                       sum(unlearning_results_patient[i][6]) / len(unlearning_results_patient[i][6]),
                       ]
            writer.writerow(content)
        writer.writerow(' ')
        writer.writerow(' ')
        writer.writerow(' ')


# SISA方法训练源模型（组合）
def SISA(num_sharded, used_dataset, used_patients, used_model, used_device,
         para_path_1, para_path_2, results_path):
    """
    SISA training method, short for Sharded, Isolated, Sliced, and Aggregate training
    :param num_sharded: int, number of sharded
    :param used_dataset:
    :param used_patients:
    :param used_model:
    :param used_device:
    :param para_path_1: sharded model parameter path
    :param para_path_2: sliced model parameter path
    :param results_path:
    :return: None
    """

    # # 患者分组（三个患者一组（大于等于2，上限可实验确定），随即分组？序列分组？）
    num_patients = len(used_patients)
    assert num_patients > num_sharded

    os.makedirs(para_path_1, exist_ok=True)
    os.makedirs(para_path_2, exist_ok=True)

    sharded_models = []

    if not os.listdir(para_path_1):
        index = 0
        for sharded_index in range(0, num_patients, num_sharded):  # 隔三个人一迭代
            # load sharded patients' data
            sharded_patients = []
            sliced_para_name = ''
            sliced_models = []
            for sliced_index in range(num_sharded):  # 每个模型使用三个患者数据，每次迭代依次增加一个患者的数据
                if index < num_patients:
                    sharded_patients.append(used_patients[index])
                    index += 1
                else:
                    break  # 最后一组不满足三个患者，直接跳出循环

                temp_name = '_'  # temp_name: 本轮循环中模型的名字, sliced_para_name: 上一轮循环中模型的名字
                for pt in sharded_patients:
                    temp_name += (pt + '_')

                print(f"get sharded patients' data {temp_name} to train models.\n")
                train_loader = load_data_balance(None, None,
                                                 used_dataset, sharded_patients)

                # get sharded models
                if sliced_index == 0:
                    sliced_model = used_model.to(used_device)
                else:
                    sliced_model = used_model.to(used_device)
                    sliced_model.load_state_dict(torch.load(
                        os.path.join(para_path_2, f'sliced_model_para{sliced_para_name}.pth')))

                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(sliced_model.parameters(), lr=0.0001)

                for epoch in range(10):
                    train_losses = []
                    sliced_model.train()
                    loop = tqdm(train_loader, ascii=True)
                    for samples, labels in loop:
                        samples = samples.to(used_device)
                        labels = labels.to(used_device)

                        optimizer.zero_grad()

                        class_preds = sliced_model(samples)
                        loss = criterion(class_preds, labels)

                        loss.backward()

                        optimizer.step()

                        train_losses.append(loss.item() / labels.shape[0])
                        loop.set_description(
                            f"Epoch: {epoch + 1}/{10} | Epoch loss: {(sum(train_losses) / len(train_losses)):.10f}"
                        )

                    loop.close()

                torch.save(sliced_model.state_dict(),
                           os.path.join(para_path_2, f'sliced_model_para{temp_name}.pth'))
                sliced_models.append(sliced_model)
                sliced_para_name = temp_name

            # 取分组模型的最后一个并保存(sharded_model)
            sharded_models.append(sliced_models[-1])
            torch.save(sliced_models[-1].state_dict(),
                       os.path.join(para_path_1, f'sharded_model_para{sliced_para_name}.pth'))

        evaluate_sharded(used_patients, sharded_models, used_device, used_dataset, results_path)

        return sharded_models

    else:
        print('load exsiting original models\n')
        sharded_models = []
        models_name = os.listdir(para_path_1)
        for name in models_name:
            sharded_model = deepcopy(used_model)
            sharded_model.load_state_dict(torch.load(os.path.join(para_path_1, name)))
            sharded_models.append(sharded_model)

        return sharded_models


def train(used_model, used_device, train_loader):

    unlearn_model = used_model.to(used_device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(unlearn_model.parameters(), lr=0.0001)

    for epoch in range(10):
        train_losses = []
        unlearn_model.train()
        loop = tqdm(train_loader, ascii=True)
        for samples, labels in loop:
            samples = samples.to(used_device)
            labels = labels.to(used_device)

            optimizer.zero_grad()

            class_preds = unlearn_model(samples)
            loss = criterion(class_preds, labels)

            loss.backward()

            optimizer.step()

            train_losses.append(loss.item() / labels.shape[0])
            loop.set_description(
                f"Epoch: {epoch + 1}/{10} | Epoch loss: {(sum(train_losses) / len(train_losses)):.10f}"
            )

        loop.close()

    return unlearn_model


def evaluate(used_dataset, unlearning_patient, remaining_patient, unlearned_model,
             original_model, device, results_path):

    unlearning_results_patient = []
    mark = []

    # test on forget data
    print('\ntesting on unlearning data')
    forget_loader = prep_test1(used_dataset, unlearning_patient)
    print('\nusing original model')
    average_metrics = test_metrics(forget_loader, original_model, device)
    unlearning_results_patient.append(average_metrics)
    mark.append(['original', 'forget', unlearning_patient[0]])
    print('using unlearned model')
    average_metrics = test_metrics(forget_loader, unlearned_model, device)
    unlearning_results_patient.append(average_metrics)
    mark.append(['unlearning', 'forget', unlearning_patient[0]])

    # test on remain data
    print('\ntesting on remaining data')
    for pat in remaining_patient:
        remain_loader = prep_test1(used_dataset, [pat])
        print('using original model')
        average_metrics = test_metrics(remain_loader, original_model, device)
        unlearning_results_patient.append(average_metrics)
        mark.append(['original', 'remain', pat])
        print('using unlearned model\n')
        average_metrics = test_metrics(remain_loader, unlearned_model, device)
        unlearning_results_patient.append(average_metrics)
        mark.append(['unlearning', 'remain', pat])

    head_mark = 0
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
                       sum(unlearning_results_patient[i+1][0]) / len(unlearning_results_patient[i+1][0]),
                       sum(unlearning_results_patient[i+1][1]) / len(unlearning_results_patient[i+1][1]),
                       sum(unlearning_results_patient[i+1][2]) / len(unlearning_results_patient[i+1][2]),
                       sum(unlearning_results_patient[i+1][3]) / len(unlearning_results_patient[i+1][3]),
                       sum(unlearning_results_patient[i+1][4]) / len(unlearning_results_patient[i+1][4]),
                       sum(unlearning_results_patient[i+1][5]) / len(unlearning_results_patient[i+1][5]),
                       sum(unlearning_results_patient[i+1][6]) / len(unlearning_results_patient[i+1][6])]
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


def SISA_unlearn(used_dataset, used_patients,  used_model, used_device,
                 para_path_1, para_path_2, results_path):
    # TODO：多患者遗忘
    sharded_models_name = os.listdir(para_path_1)
    sharded_models = []
    for models_name in sharded_models_name:
        sharded_model = deepcopy(used_model)
        sharded_model = sharded_model.to(used_device)
        sharded_model.load_state_dict(torch.load(os.path.join(para_path_1, models_name)))
        sharded_models.append(sharded_model)

    # 序列循环对患者进行遗忘
    for patient in used_patients:

        since = time.time()

        if patient in [
            # '1',  #
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
        ]:
            continue
        unlearned_models = deepcopy(sharded_models)

        remaining_patients = deepcopy(used_patients)
        remaining_patients.remove(patient)
        unlearning_patient = [patient]

        # # 找到遗忘患者对应的模型
        for model_index, sharded_model_name in enumerate(sharded_models_name):
            pt_id = sharded_model_name.split('_')[3:-1]
            len_pt = len(pt_id)
            assert len_pt >= 1
            if patient in pt_id:
                # # 找到患者的训练顺序索引
                pt_index = [index for index, item in enumerate(pt_id) if item == patient][0]  # 个分组之间独立（一个患者的数据只被使用一次）
                if pt_index == 0:  # 患者为模型训练所使用的第一个
                    if len_pt == 1:  # 如果一组中只有一个患者，直接删除对应模型
                        unlearned_models.pop(model_index)
                    elif len_pt > 1:  # 如果一组中不止一个患者，则使用后面患者的数据训练遗忘模型（这里为了方便就不再使用逐患者迭代训练方式）
                        group_patients = deepcopy(pt_id)
                        group_patients.pop(0)
                        unlearn_loader = load_data_balance(None, None,
                                                           used_dataset, group_patients)
                        unlearn_model = deepcopy(used_model)
                        unlearned_model = train(unlearn_model, used_device, unlearn_loader)
                        unlearned_models.pop(model_index)
                        unlearned_models.append(unlearned_model)
                    else:
                        raise ValueError
                elif 0 < pt_index < len_pt - 1:  # 使用遗忘患者之后的患者数据在遗忘患者之前的模型基础上进行遗忘训练
                    group_patients = deepcopy(pt_id)
                    before_model_name = '_'
                    while pt_index + 1:  # +1 索引从零开始的
                        if pt_index >= 1:  # 遗忘患者之前的模型名
                            before_model_name += group_patients.pop(0) + '_'
                        else:  # 不使用遗忘患者的数据
                            group_patients.pop(0)
                        pt_index -= 1
                    unlearn_loader = load_data_balance(None, None,
                                                       used_dataset, group_patients)
                    before_model = deepcopy(used_model)
                    before_model_name = 'sliced_model_para' + before_model_name + '.pth'
                    before_model.load_state_dict(torch.load(os.path.join(para_path_2, before_model_name)))

                    unlearned_model = train(before_model, used_device, unlearn_loader)
                    unlearned_models.pop(model_index)
                    unlearned_models.append(unlearned_model)
                elif pt_index == len_pt - 1:  # 遗忘患者为最后每组中的最后一个，使用倒数第二个模型作为遗忘模型
                    unlearned_models.pop(model_index)
                    unlearned_model = deepcopy(used_model)
                    group_patients = deepcopy(pt_id)
                    group_patients.pop()  # 将最后一个患者删除
                    pt_name = '_'
                    for i in group_patients:
                        pt_name += i + '_'
                    unlearned_model_name = f'sliced_model_para{pt_name}.pth'
                    unlearned_model.load_state_dict(
                        torch.load(os.path.join(para_path_2, unlearned_model_name)))
                    unlearned_models.append(unlearned_model)
                else:
                    raise ValueError
            else:
                continue

            end = time.time()
            print(f"time of unlearning patient {patient} is {(end-since)/3600} s")

            evaluate(used_dataset, unlearning_patient, remaining_patients, unlearned_models, sharded_models,
                     used_device, results_path)

    print("SISA_unlearn()")


if __name__ == "__main__":

    dataset = ['CHBMIT', 'Kaggle2014Pred']
    used_dataset_ = dataset[0]

    models = {'CHBMIT': {'MLP': STMLP(**configs["ST-MLP"]), 'ViT': ViT(), 'PCT': PCT(), 'CNN': CNN()},
              'Kaggle2014Pred': {'MLP': STMLP(**configs["ST-MLP"]), 'ViT': ViT_kaggle(), 'PCT': PCT_kaggle(),
                                 'CNN': CNN_kaggle()}}

    model_name = 'PCT'
    used_model_ = models[used_dataset_][model_name]

    unlearn_methods_ = ['SISA']
    used_method_ = unlearn_methods_[0]

    para_pathes = {'sharded_path': f'models/{used_method_}/sharded_model/{used_dataset_}/{model_name}',
                   'sliced_path': f'models/{used_method_}/sliced_model/{used_dataset_}/{model_name}',
                   'unlearn_path': f'models/{used_method_}/unlearned_model/{used_dataset_}/{model_name}'}
    os.makedirs(para_pathes['sharded_path'], exist_ok=True)
    os.makedirs(para_pathes['sliced_path'], exist_ok=True)
    os.makedirs(para_pathes['unlearn_path'], exist_ok=True)

    results_path_ = {'sharded': f'results/sharded/{used_dataset_}/{model_name}',
                     'unlearn': f'results/unlearn/{used_method_}/{used_dataset_}/{model_name}'}
    os.makedirs(results_path_['sharded'], exist_ok=True)
    os.makedirs(results_path_['unlearn'], exist_ok=True)

    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    all_patients, skip_patients = [], []
    if used_dataset_ == 'CHBMIT':

        all_patients = [
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

        all_patients = [
            'Dog_1',
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

    num_sharded_ = 3

    print("SISA training")
    SISA(num_sharded_, used_dataset_, all_patients, used_model_, device_,
         para_pathes['sharded_path'], para_pathes['sliced_path'], results_path_['sharded'])

    print("\nSISA unlearning")
    SISA_unlearn(used_dataset_, all_patients, used_model_, device_,
                 para_pathes['sharded_path'], para_pathes['sliced_path'], results_path_['unlearn'])

    print("comprison method_SISA.py")
