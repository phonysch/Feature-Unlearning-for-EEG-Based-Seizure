import os
import csv
from copy import deepcopy
import torch
from PCT_net import PCT
from PCT_net_kaggle import PCT as PCT_kaggle
from CNN import CNN_kaggle
from CNN import CNN
from tqdm import tqdm
from STMLP import STMLP
from Transformer import ViT
from Transformer import ViT_kaggle
from STMLP_configs import configs
from unlearning_scheme import shadow_states, influence_function, retrain_featurer, retrain_classifier, \
    randomly_initialize, retrain_featurer_only_remaining
from functions import load_data, test_metrics, prep_test1, load_data_influence, load_data_balance


def evaluate(dataset, unlearning_patient, remaining_patient, unlearned_model, original_model, device, results_path):

    unlearning_results_patient = []
    mark = []

    # test on forget data
    print('\ntesting on unlearning data')
    forget_loader = prep_test1(dataset, unlearning_patient)
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
        remain_loader = prep_test1(dataset, [pat])
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


def evaluate_original(patients, model, device, dataset, save_path):

    unlearning_results_patient = []
    mark = []

    # test on remain data
    print('\ntesting on remaining data')
    for pat in patients:
        remain_loader = prep_test1(dataset, [pat])
        print('using original model')
        average_metrics = test_metrics(remain_loader, model, device)
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


def train_original_model(original_patients, model, device, lr, epochs, para_path, results_path, dataset):

    os.makedirs(para_path, exist_ok=True)
    if not os.path.exists(os.path.join(para_path, 'original_model_para.pth')):
        # load original patients' data
        print("get all patients' data to train an original model.\n")
        train_loader = load_data_balance(None, None, dataset, original_patients)

        # get original model
        original_model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(original_model.parameters(), lr=lr)

        print('train original model\n')
        for epoch in range(epochs):
            train_losses = []
            original_model.train()
            loop = tqdm(train_loader, ascii=True)
            for samples, labels in loop:
                samples = samples.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                class_preds = original_model(samples)
                loss = criterion(class_preds, labels)

                loss.backward()

                optimizer.step()

                train_losses.append(loss.item() / labels.shape[0])
                loop.set_description(
                    f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(train_losses) / len(train_losses)):.10f}"
                )

            loop.close()

        torch.save(original_model.state_dict(), os.path.join(para_path, 'original_model_para.pth'))
        evaluate_original(original_patients, original_model, device, dataset, results_path)

        return original_model

    else:
        print('load exsiting original models\n')
        original_model = deepcopy(model)
        original_model.load_state_dict(torch.load(os.path.join(para_path, 'original_model_para.pth')))
        return original_model


def unlearning_methods(dataset, remaining_patients, unlearning_patient, model, device, lr, epochs, path,
                       unlearn_methods, influence_methods, shadow_methods, loss_rate):
    """
    training unlearning models,
    :param dataset:
    :param remaining_patients: str in list, remaining patients' id
    :param unlearning_patient: str in list, unlearning patient's id
    :param model: network architecture
    :param device: cpu or gpu (default: gpu)
    :param lr: float in list, [0]: repairing learning rate; [1] unlearning learning rate
    :param epochs:
    :param path: str in list, [0]: original model parameter path; [1]: unlearning model parameter path
    :param unlearn_methods:
    :param influence_methods: methods for influence functions to lighten the influence of unlearning data
    :param shadow_methods: methods for shadow states to lighten the influence of unlearning data
    :param loss_rate: folat in list; alpha, beta, gamma
    :return: unlearning model
    """

    # unlearning
    # # shadow states
    if unlearn_methods == 'shadow states':
        print('unlearning method: shadow states\n')

        if not os.path.exists(os.path.join(path['unlearn_path'], f'unlearned_model_para_{unlearning_patient[0]}.pth')):
            # # load partial remaining patients (with a rate of 0.1) and unlearning patient data
            unlearning_loader = load_data_balance(True, 2, dataset, remaining_patients, unlearning_patient)

            extract_states = 0
            unlearned_model = shadow_states(model, unlearning_loader, device, lr, epochs, path,
                                            extract_states, shadow_methods)
        else:
            unlearned_model = \
                model.load_state_dict(torch.load(os.path.join(path['unlearn_path'],
                                                              f'unlearned_model_para_{unlearning_patient[0]}.pth')))
        return unlearned_model

    # # influence functions
    elif unlearn_methods == 'influence functions':
        print('unlearning method: influence functions\n')
        # #
        loader_subset_remaining, loader_forgetting_raw, loader_forgetting_disturb = \
            load_data_influence(influence_methods, remaining_patients, unlearning_patient)

        if not os.path.exists(os.path.join(path['unlearn_path'], f'unlearned_model_para_{unlearning_patient[0]}.pth')):
            unlearned_model = \
                influence_function(model,
                                   loader_subset_remaining, loader_forgetting_raw, loader_forgetting_disturb,
                                   order=1, device=device, lr=lr, epochs=epochs, path=path)
        else:
            unlearned_model = \
                model.load_state_dict(torch.load(os.path.join(path['unlearn_path'],
                                                              f'unlearned_model_para_{unlearning_patient[0]}.pth')))
        return unlearned_model

    elif unlearn_methods == 'retrain featurer':
        print('unlearning method: retrain featurer\n')

        only_remaining = False

        if only_remaining:  # only use remaining samples in the unlearning phase (without use unlearning samples)
            if not os.path.exists(os.path.join(path['unlearn_path'],
                                               f'unlearned_model_para_{unlearning_patient[0]}.pth')):
                # 0: augmentation method, 1: perturbation method, 2: original data without any method
                augmentation_perturbation = 2
                unlearning_loader = load_data_balance(False, augmentation_perturbation, dataset,
                                                      remaining_patients, unlearning_patient)

                unlearning_model = deepcopy(model)
                unlearning_model.load_state_dict(torch.load(os.path.join(path['original_path'],
                                                                         'original_model_para.pth')))

                retrain_method = ['retrain', 'change', 'complementary', 'reverse']
                used_method = retrain_method[3]

                unlearned_model = retrain_featurer_only_remaining(unlearning_model, unlearning_loader,
                                                                  device, lr[0], epochs,
                                                                  used_method, augmentation_perturbation)
            else:
                unlearned_model = \
                    model.load_state_dict(torch.load(os.path.join(path['unlearn_path'],
                                                                  f'unlearned_model_para_{unlearning_patient[0]}.pth')))

            return unlearned_model
        else:
            if not os.path.exists(os.path.join(path['unlearn_path'],
                                               f'unlearned_model_para_{unlearning_patient[0]}.pth')):
                # 0: augmentation method, 1: perturbation method, 2: original data without any method
                augmentation_perturbation = 2
                unlearning_loader = load_data_balance(True, augmentation_perturbation, dataset,
                                                      remaining_patients, unlearning_patient)

                unlearning_model = deepcopy(model)
                unlearning_model.load_state_dict(torch.load(os.path.join(path['original_path'],
                                                                         'original_model_para.pth')))

                retrain_method = ['retrain', 'change', 'complementary', 'reverse']
                used_method = retrain_method[3]

                unlearned_model = retrain_featurer(unlearning_model, unlearning_loader, device, lr[0], epochs,
                                                   used_method, augmentation_perturbation, loss_rate)
            else:
                unlearned_model = \
                    model.load_state_dict(torch.load(os.path.join(path['unlearn_path'],
                                                                  f'unlearned_model_para_{unlearning_patient[0]}.pth')))

            return unlearned_model

    elif unlearn_methods == 'retrain classifier':
        print('unlearning method: retrain classifier\n')

        if not os.path.exists(os.path.join(path['unlearn_path'], f'unlearned_model_para_{unlearning_patient[0]}.pth')):
            unlearning_loader = load_data(True, remaining_patients, unlearning_patient)

            unlearning_model = deepcopy(model)
            unlearning_model.load_state_dict(torch.load(os.path.join(path['original_path'], 'original_model_para.pth')))

            retrain_method = ['retrain', 'change', 'complementary']

            unlearned_model = retrain_classifier(unlearning_model, unlearning_loader, device, lr[0], epochs,
                                                 retrain_method[2])
        else:
            unlearned_model = \
                model.load_state_dict(torch.load(os.path.join(path['unlearn_path'],
                                                              f'unlearned_model_para_{unlearning_patient[0]}.pth')))

        return unlearned_model

    elif unlearn_methods == 'randomly initialize':
        print('unlearning method: randomly initialize\n')

        if not os.path.exists(os.path.join(path['unlearn_path'], f'unlearned_model_para_{unlearning_patient[0]}.pth')):
            # 0: augmentation method, 1: perturbation method, 2: original data without any method
            augmentation_perturbation = 0
            unlearning_loader = load_data_balance(True, augmentation_perturbation, dataset,
                                                  remaining_patients, unlearning_patient)

            original_model = deepcopy(model)
            raw_model = deepcopy(model)
            original_model.load_state_dict(torch.load(os.path.join(path['original_path'], 'original_model_para.pth')))

            unlearned_model = randomly_initialize(original_model, raw_model, unlearning_loader, device, lr[0], epochs,
                                                  augmentation_perturbation)
        else:
            unlearned_model = \
                model.load_state_dict(torch.load(os.path.join(path['unlearn_path'],
                                                              f'unlearned_model_para_{unlearning_patient[0]}.pth')))

        return unlearned_model

    else:
        print('unexcept unlearning methods')
        exit(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # # automatic
    # dataset_ = ['CHBMIT', 'Kaggle2014Pred']
    #
    # models = {'CHBMIT': {'MLP': STMLP(**configs["ST-MLP"]), 'ViT': ViT(), 'PCT': PCT(), 'CNN': CNN()},
    #           'Kaggle2014Pred': {'MLP': STMLP(**configs["ST-MLP_K"]), 'ViT': ViT_kaggle(), 'PCT': PCT_kaggle(),
    #                              'CNN': CNN_kaggle()}}
    #
    # for d in models:
    #     if d == 'Kaggle2014Pred':
    #         continue
    #     used_dataset_ = d
    #     for model_ in models[d]:
    #         if model_ == 'PCT':
    #             continue
    #
    #         print(model_)
    #
    #         model_name = model_
    #         used_model = models[d][model_name]
    #
    #         unlearn_methods_ = ['shadow states', 'influence functions', 'retrain featurer', 'retrain classifier',
    #                             'randomly initialize']
    #         used_method_ = unlearn_methods_[2]
    #
    #         para_pathes = {'original_path': f'models/{used_method_}/original_model/{used_dataset_}/{model_name}',
    #                        'unlearn_path': f'models/{used_method_}/unlearned_model/{used_dataset_}/{model_name}'}
    #         os.makedirs(para_pathes['original_path'], exist_ok=True)
    #         os.makedirs(para_pathes['unlearn_path'], exist_ok=True)
    #
    #         results_path_ = {'original': f'results/original/{used_dataset_}/{model_name}',
    #                          'unlearn': f'results/unlearn/{used_method_}/{used_dataset_}/{model_name}'}
    #         os.makedirs(results_path_['original'], exist_ok=True)
    #         os.makedirs(results_path_['unlearn'], exist_ok=True)
    #
    #         device_ = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #
    #         all_patients, skip_patients = [], []
    #         if used_dataset_ == 'CHBMIT':
    #
    #             all_patients = [
    #                     '1',  #
    #                     '2',  #
    #                     '3',  #
    #                     '5',  #
    #                     '6',  #
    #                     '8',  #
    #                     '9',  #
    #                     '10',  #
    #                     '13',  #
    #                     '14',  #
    #                     '16',  #
    #                     '17',  #
    #                     '18',  #
    #                     '19',  #
    #                     '20',  #
    #                     '21',  #
    #                     '22',  #
    #                     '23'  #
    #                 ]
    #             skip_patients = [
    #                     # '1',
    #                     # '2',
    #                     # '3',
    #                     # '5',
    #                     # '6',
    #                     # '8',
    #                     # '9',
    #                     # '10',
    #                     # '13',
    #                     # '14',
    #                     # '16',
    #                     # '17',
    #                     # '18',
    #                     # '19',
    #                     # '20',
    #                     # '21',
    #                     # '22',
    #                     # '23'
    #                 ]
    #
    #         elif used_dataset_ == 'Kaggle2014Pred':
    #
    #             all_patients = [
    #                 'Dog_1',
    #                 'Dog_2',
    #                 'Dog_3',
    #                 'Dog_4',
    #                 'Dog_5',
    #                 # 'Patient_1',
    #                 # 'Patient_2',
    #             ]
    #             skip_patients = [
    #                 # 'Dog_1',
    #                 # 'Dog_2',
    #                 # 'Dog_3',
    #                 # 'Dog_4',
    #                 # 'Dog_5',
    #                 # 'Patient_1',
    #                 # 'Patient_2',
    #             ]
    #
    #         else:
    #
    #             print('unexcepted dataset')
    #             exit()
    #
    #         # settings for training orginal models
    #         lr_o = 0.0001
    #         epoch_o = 10
    #         original_model_ = train_original_model(all_patients, used_model, device_, lr_o, epoch_o,
    #                                                para_pathes['original_path'], results_path_['original'],
    #                                                used_dataset_)
    #
    #         # settings for training unlearning models
    #         lr_u = [0.0001, 0.00001]
    #         epoch_u = 1
    #         disturb_methods_influence = ['samples', 'labels', 'samples_labels']
    #         label_methods_shadow = ['reverse', 'complementary']
    #         for patient in all_patients:
    #             # # skip some patients (patients whose data have already unlearned)
    #             if patient in skip_patients:
    #                 print(f'skip case: {patient}\n')
    #                 continue
    #
    #             unlearning_patient_ = [patient]
    #             remaining_patients_ = deepcopy(all_patients)
    #             remaining_patients_.remove(patient)
    #
    #             # unlearn_methods_ = ['shadow states', 'influence functions', 'retrain calssifier']
    #             # disturb_methods_ = ['samples', 'labels', 'samples_labels']
    #             print(f'train unlearning models, unlearning case: {patient}\n')
    #             unlearned_model_ = \
    #                 unlearning_methods(used_dataset_, remaining_patients_, unlearning_patient_,
    #                                    used_model, device_, lr_u, epoch_u, para_pathes,
    #                                    used_method_, disturb_methods_influence[0], label_methods_shadow[1])
    #
    #             print(f'evaluate the unlearned model of case: {patient}')
    #             evaluate(used_dataset_, unlearning_patient_, remaining_patients_, unlearned_model_, original_model_,
    #                      device_, results_path_['unlearn'])

    # handle
    dataset_ = ['CHBMIT', 'Kaggle2014Pred']
    used_dataset_ = dataset_[0]

    models = {'CHBMIT': {'MLP': STMLP(**configs["ST-MLP"]), 'ViT': ViT(), 'PCT': PCT(), 'CNN': CNN()},
              'Kaggle2014Pred': {'MLP': STMLP(**configs["ST-MLP"]), 'ViT': ViT_kaggle(), 'PCT': PCT_kaggle(),
                                 'CNN': CNN_kaggle()}}

    model_name = 'PCT'
    used_model = models[used_dataset_][model_name]

    unlearn_methods_ = ['shadow states', 'influence functions', 'retrain featurer', 'retrain classifier',
                        'randomly initialize']
    used_method_ = unlearn_methods_[2]

    para_pathes = {'original_path': f'models/{used_method_}/original_model/{used_dataset_}/{model_name}',
                   'unlearn_path': f'models/{used_method_}/unlearned_model/{used_dataset_}/{model_name}'}
    os.makedirs(para_pathes['original_path'], exist_ok=True)
    os.makedirs(para_pathes['unlearn_path'], exist_ok=True)

    results_path_ = {'original': f'results/original/{used_dataset_}/{model_name}',
                     'unlearn': f'results/unlearn/{used_method_}/{used_dataset_}/{model_name}'}
    os.makedirs(results_path_['original'], exist_ok=True)
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

    # settings for training orginal models
    lr_o = 0.0001
    epoch_o = 5
    original_model_ = train_original_model(all_patients, used_model, device_, lr_o, epoch_o,
                                           para_pathes['original_path'], results_path_['original'],
                                           used_dataset_)

    # settings for training unlearning models
    lr_u = [0.0001, 0.0001]
    epoch_u = 10
    disturb_methods_influence = ['samples', 'labels', 'samples_labels']
    label_methods_shadow = ['reverse', 'complementary']

    parameter_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    alpha = beta = gamma = 0.0
    for i_o in range(1, 4):
        for v in parameter_list:
            if i_o == 1:
                alpha = v
                beta = gamma = (1 - alpha) / 2
            elif i_o == 2:
                beta = v
                alpha = gamma = (1 - beta) / 2
            elif i_o == 3:
                gamma = v
                alpha = beta = (1 - gamma) / 2
            else:
                raise ValueError
            loss_rate_ = [alpha, beta, gamma]

            for patient in all_patients:
                # # skip some patients (patients whose data have already unlearned)
                if patient in skip_patients:
                    print(f'skip case: {patient}\n')
                    continue

                unlearning_patient_ = [patient]
                remaining_patients_ = deepcopy(all_patients)
                remaining_patients_.remove(patient)

                # unlearn_methods_ = ['shadow states', 'influence functions', 'retrain calssifier']
                # disturb_methods_ = ['samples', 'labels', 'samples_labels']
                print(f'train unlearning models, unlearning case: {patient}\n')
                unlearned_model_ = \
                    unlearning_methods(used_dataset_, remaining_patients_, unlearning_patient_,
                                       used_model, device_, lr_u, epoch_u, para_pathes,
                                       used_method_, disturb_methods_influence[0], label_methods_shadow[1],
                                       loss_rate_)

                print(f'evaluate the unlearned model of case: {patient}')
                evaluate(used_dataset_, unlearning_patient_, remaining_patients_, unlearned_model_, original_model_,
                         device_, results_path_['unlearn'])

    print('main')
