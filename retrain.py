import os
import csv
from copy import deepcopy
import torch
# from PCT_net import PCT
# from PCT_net_original import PCT
from PCT_net_test import PCT
from CNN import CNN
from tqdm import tqdm
from STMLP import STMLP
from Transformer import ViT
from STMLP_configs import configs
from functions import load_data, test_metrics, prep_test1, test_metrics_records_retrain
import time
from RepNet import RepNet


def evaluate(excepted_patient, remaining_patient, retrained_model, device, results_path, dataset):

    unlearning_results_patient = []
    mark = []

    # test on forget data
    print('\ntesting on unlearning data')
    forget_loader = prep_test1(dataset, excepted_patient)
    average_metrics = test_metrics_records_retrain(forget_loader, retrained_model, device)
    unlearning_results_patient.append(average_metrics)
    mark.append(['excepted', excepted_patient[0]])

    # test on remain data
    print('\ntesting on remaining data')
    for pat in remaining_patient:
        remain_loader = prep_test1(dataset, [pat])
        average_metrics = test_metrics(remain_loader, retrained_model, device)
        unlearning_results_patient.append(average_metrics)
        mark.append(['remaining', pat])

    save_path = results_path
    os.makedirs(save_path, exist_ok=True)
    head_mark = 0
    with open(os.path.join(save_path, 'results.csv'), 'a+', encoding='utf8', newline='') as file:
        writer = csv.writer(file)
        if head_mark == 0:
            writer.writerow([' ', ' ', 'Sn', 'Sp', 'precision', 'acc', 'f1_score', 'auc', 'fpr'])
            head_mark += 1
        for i in range(len(unlearning_results_patient)):
            content = [mark[i][0], mark[i][1],
                       sum(unlearning_results_patient[i][0]) / len(unlearning_results_patient[i][0]),
                       sum(unlearning_results_patient[i][1]) / len(unlearning_results_patient[i][1]),
                       sum(unlearning_results_patient[i][2]) / len(unlearning_results_patient[i][2]),
                       sum(unlearning_results_patient[i][3]) / len(unlearning_results_patient[i][3]),
                       sum(unlearning_results_patient[i][4]) / len(unlearning_results_patient[i][4]),
                       sum(unlearning_results_patient[i][5]) / len(unlearning_results_patient[i][5]),
                       sum(unlearning_results_patient[i][6]) / len(unlearning_results_patient[i][6])]
            writer.writerow(content)
            if i == 0:
                writer.writerow(' ')
            else:
                pass
        writer.writerow(' ')
        writer.writerow(' ')
        writer.writerow(' ')


def retrain(remaining_patients, model, device, lr, epochs):

    retrain_loader = load_data(False, remaining_patients)
    retrain_model = deepcopy(model)

    since = time.time()
    optimizer = torch.optim.Adam(retrain_model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()
    retrain_model.to(device)
    retrain_model.train()
    for epoch in range(epochs):
        train_losses = []
        loop = tqdm(retrain_loader)
        for samples, labels in loop:
            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = retrain_model(samples)

            loss = criterion(outs, labels)
            loss.backward()

            optimizer.step()

            train_losses.append(loss.item() / labels.shape[0])
            loop.set_description(
                f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(train_losses) / len(train_losses)):.10f}"
            )

        loop.close()

    stop = time.time()
    print(f'retrain time consuming: {(stop-since):.2f} s')

    return retrain_model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    dataset_ = ['CHBMIT', 'Kaggle2014Pred']
    used_dataset_ = dataset_[0]

    all_patients = [
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
    skip_patients = [
        '1',  #
        # '2',
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

    models = {'MLP': STMLP(**configs["ST-MLP"]), 'ViT': ViT(), 'PCT': PCT(), 'CNN': CNN(), 'RepNet': RepNet(22)}
    model_name = 'PCT'
    used_model = models[model_name]

    results_save_path_ = f'results/retrain/{model_name}'
    para_save_path_ = f'models/retrain/{model_name}'

    os.makedirs(results_save_path_, exist_ok=True)
    os.makedirs(para_save_path_, exist_ok=True)

    device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lr_ = 0.0001
    epochs_ = 10

    for patient in all_patients:
        if patient in skip_patients:
            print(f'skip patient: {patient}\n')
            continue

        excepted_patient_ = [patient]
        remaining_patients_ = deepcopy(all_patients)
        remaining_patients_.remove(patient)

        print(f'\nretraining from scratch, retrain without patient: {patient}\n')
        if not os.path.exists(os.path.join(para_save_path_, f'except_{patient}_para.pth')):

            retrained_model_ = retrain(remaining_patients_, used_model, device_, lr_, epochs_)

            torch.save(retrained_model_.state_dict(), os.path.join(para_save_path_, f'except_{patient}_para.pth'))

        else:
            print('get existing retrained model')
            retrained_model_ = deepcopy(used_model)
            retrained_model_.load_state_dict(torch.load(os.path.join(para_save_path_,
                                                                     f'except_{patient}_para.pth')))

        print('evaluate on excepted patient and remaining patients')
        evaluate(excepted_patient_, remaining_patients_, retrained_model_, device_, results_save_path_, used_dataset_)

    print('main')
