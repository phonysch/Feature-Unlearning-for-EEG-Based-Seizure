from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as func
import torch
import os
from unlearning_scheme_functions import get_gradients, update_model_parameters, separate_prediction, \
    generate_complementary_labels, complementary_loss_choose, loss_Tsallis, augmentation_perturbation
import time


def shadow_states(model, loader, device, lr, epochs, path, extra_states, label_methods):
    """
    forget specific patient's data
    :param model: model, unloaded parameters
    :param loader: tensor in dataloader, [0]: samples; [1]: class labels;
                   [2]: domain labels ('0' for remaining samples, '1' for unlearning samples.)
    :param device:
    :param lr: float in list, [0]: repairing learning rate; [1] unlearning learning rate
    :param epochs:
    :param path: [1] original model parameters, [2] unlearning models parameters.
    :param extra_states: '2' and '3' for specific patient's preictal and interictal, respectively.
    :param label_methods:
    :return: unlearned model
    """
    unlearning_model = deepcopy(model)
    unlearning_model.load_state_dict(torch.load(os.path.join(path['original_path'], 'original_model_para.pth')))

    # extra states
    if extra_states != 0:  # use this method should change line 138 and 139 in "load_data" in functions.py
        unlearning_model = extra_states_method(unlearning_model, loader, device, lr, epochs, extra_states)
    else:
        if label_methods == 'reverse':
            unlearning_model = reverse_labels_method(unlearning_model, loader, device, lr, epochs)
        elif label_methods == 'complementary':
            unlearning_model = complementary_labels_method(model, loader, device, lr, epochs)

    return unlearning_model


def influence_function(model, loader_subset_remaining, loader_forgetting_raw, loader_forgetting_disturb,
                       order, device, lr, epochs, path):
    """
    unlearning data with influence functions which based on first-order and second-order differentiation, the arguments
    'loader_forgetting_raw' and
    :param model: network architecture
    :param loader_subset_remaining: subset of remaining patients' data
    :param loader_forgetting_raw: forgetting patients' data without processing.
    :param loader_forgetting_disturb: forgetting patients' data with processing (one of reversing labels,
                                            random labels or samples processing)
    :param order: 1: for using first-order-updating strategy, 2: for using second-order-updating strategy.
    :param device: cpu or gpu
    :param lr:
    :param epochs:
    :param path: [0]: original model parameter path (for loading), [1]: unlearned model parameter path (for saving)
    :return: unlearned model.
    """
    # load parameters
    unlearning_model = deepcopy(model)
    unlearning_model.load_state_dict(torch.load(os.path.join(path['original_path'], 'original_model_para.pth')))

    # unlearning (two methods to update model parameters, choose one of the methods (comment the other methods))
    unlearning_model.to(device)
    if order == 1:
        # # # update model parameters iteratively
        # optimizer = optim.Adam(unlearning_model.parameters(), lr=lr[1])
        # criterion = nn.CrossEntropyLoss()
        # for epoch in range(epochs):
        #     loop = tqdm(loader_forgetting_raw, ascii=True)
        #     train_losses = []
        #     unlearning_model.train()
        #     loader_disturb = iter(loader_forgetting_disturb)
        #     print("deleting forgetting patient's data information")
        #     for samples_f_r, labels_f_r in loop:
        #         samples_f_d, labels_f_d = next(loader_disturb)
        #         samples_f_d = samples_f_d.to(device)
        #         labels_f_d = labels_f_d.to(device)
        #         samples_f_r = samples_f_r.to(device)
        #         labels_f_r = labels_f_r.to(device)
        #
        #         optimizer.zero_grad()
        #
        #         logits_f_r = unlearning_model(samples_f_r)
        #         loss_f_r = criterion(logits_f_r, labels_f_r)
        #         logits_f_d = unlearning_model(samples_f_d)
        #         loss_f_d = criterion(logits_f_d, labels_f_d)
        #
        #         loss = loss_f_d - loss_f_r
        #
        #         loss.backward()
        #         optimizer.step()
        #
        #         train_losses.append(loss.item() / labels_f_r.shape[0])
        #         loop.set_description(
        #             f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(train_losses) / len(train_losses)):.10f}"
        #         )
        #
        #     loop.close()
        #
        #     # # repairing the information of remaining patients.
        #     print("repairing remaining patients' data information")
        #     unlearning_model = repairing_train(unlearning_model, loader_subset_remaining, device,
        #                                        repairing_lr=lr[0], repairing_epoch=1)

        # # update all model parameters at once
        update_times = 2
        for times in range(update_times):
            gradients = get_gradients(unlearning_model, loader_forgetting_raw, loader_forgetting_disturb, lr, device)
            unlearning_model = update_model_parameters(unlearning_model, gradients, 0.0001)
            # unlearning_model = repairing_train(unlearning_model, loader_subset_remaining, device,
            #                                    repairing_lr=0.0001, repairing_epoch=1)

    elif order == 2:

        print('second-order-update')
    else:
        print('unexcept input')
        exit(0)

    return unlearning_model


def retrain_featurer(original_model, unlearning_loader, device, lr, epochs, method, aug_perturb,
                     loss_rate):
    """

    :param original_model:
    :param unlearning_loader:
    :param device:
    :param lr:
    :param epochs:
    :param method: str in list, 'retrain': only update parameters in the clssifier,
                                'change': change the structure of the classifier.
    :param aug_perturb:
    :param loss_rate:
    :return:
    """

    # retrain classifier
    # # update classifier parameters
    if method == 'retrain':
        unlearning_model = deepcopy(original_model)

        opimizer = optim.Adam(unlearning_model.featurer.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        unlearning_model.to(device)
        unlearning_model.train()
        for param in unlearning_model.classifier.parameters():
            param.required_grad = False

        for epoch in range(epochs):
            loop = tqdm(unlearning_loader, ascii=True)
            training_loss = []
            for samples, labels, labels_domain in loop:
                samples = samples.to(device)
                labels = labels.to(device)

                opimizer.zero_grad()

                logits = unlearning_model(samples)
                remaining_logits, unlearning_logits, remaining_labels, unlearning_labels = \
                    separate_prediction([labels], labels_domain, device, logits)

                loss_remaining = criterion(remaining_logits, remaining_labels)
                loss_unlearning = criterion(unlearning_logits, unlearning_labels)
                alpha = 1.0
                beta = 1.0
                loss = alpha*loss_remaining + beta*loss_unlearning
                loss.backward()

                opimizer.step()

                training_loss.append(loss.item() / labels.shape[0])
                loop.set_description(
                    f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
                )

            loop.close()

        return unlearning_model

    # # change classifier
    elif method == 'change':
        print('code not available')

    elif method == 'complementary':

        unlearning_model = retrain_featurer_complementary_method(original_model, unlearning_loader, device, lr, epochs,
                                                                 aug_perturb)

        return unlearning_model

    elif method == 'reverse':
        # retrain classifier
        # # update classifier parameters
        unlearning_model = deepcopy(original_model)

        since = time.time()
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
                alpha = loss_rate[0]
                beta = loss_rate[1]
                gamma = loss_rate[2]

                loss = alpha * loss_unlearn + beta * loss_remain + gamma * loss_T
                # loss = alpha * loss_unlearn + beta * loss_remain
                loss.backward()

                optimizer.step()

                training_loss.append(loss.item() / labels_class.shape[0])
                loop.set_description(
                    f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
                )

            loop.close()

        stop = time.time()

        print(f'unlearning time consume: {(stop-since):.2f} s')

        return unlearning_model

    else:
        print('unexpected method')
        exit()

    print('retrain featurer')


def retrain_featurer_only_remaining(original_model, unlearning_loader, device, lr, epochs, method, aug_perturb):
    """

    :param original_model:
    :param unlearning_loader:
    :param device:
    :param lr:
    :param epochs:
    :param method: str in list, 'retrain': only update parameters in the clssifier,
                                'change': change the structure of the classifier.
    :param aug_perturb:
    :return:
    """

    # retrain classifier
    # # update classifier parameters
    if method == 'retrain':
        unlearning_model = deepcopy(original_model)

        opimizer = optim.Adam(unlearning_model.featurer.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        unlearning_model.to(device)
        unlearning_model.train()
        for param in unlearning_model.classifier.parameters():
            param.required_grad = False

        for epoch in range(epochs):
            loop = tqdm(unlearning_loader, ascii=True)
            training_loss = []
            for samples, labels, labels_domain in loop:
                samples = samples.to(device)
                labels = labels.to(device)

                opimizer.zero_grad()

                logits = unlearning_model(samples)
                remaining_logits, unlearning_logits, remaining_labels, unlearning_labels = \
                    separate_prediction([labels], labels_domain, device, logits)

                loss_remaining = criterion(remaining_logits, remaining_labels)
                loss_unlearning = criterion(unlearning_logits, unlearning_labels)
                alpha = 1.0
                beta = 1.0
                loss = alpha*loss_remaining + beta*loss_unlearning
                loss.backward()

                opimizer.step()

                training_loss.append(loss.item() / labels.shape[0])
                loop.set_description(
                    f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
                )

            loop.close()

        return unlearning_model

    # # change classifier
    elif method == 'change':
        print('code not available')

    elif method == 'complementary':

        unlearning_model = retrain_featurer_complementary_method(original_model, unlearning_loader, device, lr, epochs,
                                                                 aug_perturb)

        return unlearning_model

    elif method == 'reverse':
        # retrain classifier
        # # update classifier parameters
        unlearning_model = deepcopy(original_model)

        optimizer = optim.Adam(unlearning_model.featurer.parameters(), lr=lr)
        # criterion_ce = nn.CrossEntropyLoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        original_model.to(device)
        original_model.eval()
        unlearning_model.to(device)
        unlearning_model.train()
        for param in unlearning_model.classifier.parameters():
            param.required_grad = False

        for epoch in range(epochs):
            loop = tqdm(unlearning_loader, ascii=True)
            training_loss = []
            for samples, labels_class in loop:

                samples = samples.to(device)
                labels_class = labels_class.to(device)

                optimizer.zero_grad()
                dynamic_logits = unlearning_model(samples)
                static_logits = original_model(samples)

                loss_remain = criterion_kl(torch.log(func.softmax(dynamic_logits, dim=-1)),
                                           func.softmax(static_logits, dim=-1))
                loss_T = loss_Tsallis(dynamic_logits)

                beta = 1.0
                gamma = 0.0

                loss = beta * loss_remain + gamma * loss_T
                loss.backward()

                optimizer.step()

                training_loss.append(loss.item() / labels_class.shape[0])
                loop.set_description(
                    f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
                )

            loop.close()

        return unlearning_model

    else:
        print('unexpected method')
        exit()

    print('retrain featurer')


def retrain_classifier(original_model, unlearning_loader, device, lr, epochs, method):
    """

    :param original_model:
    :param unlearning_loader:
    :param device:
    :param lr:
    :param epochs:
    :param method: str in list, 'retrain': only update parameters in the clssifier,
                                'change': change the structure of the classifier.
    :return:
    """

    # retrain classifier
    # # update classifier parameters
    if method == 'retrain':
        unlearning_model = deepcopy(original_model)

        opimizer = optim.Adam(unlearning_model.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        unlearning_model.to(device)
        unlearning_model.train()
        for param in unlearning_model.featurer.parameters():
            param.required_grad = False

        for epoch in range(epochs):
            loop = tqdm(unlearning_loader, ascii=True)
            training_loss = []
            for samples, labels, labels_domain in loop:
                samples = samples.to(device)
                labels = labels.to(device)

                opimizer.zero_grad()

                logits = unlearning_model(samples)
                remaining_logits, unlearning_logits, remaining_labels, unlearning_labels = \
                    separate_prediction([labels], labels_domain, device, logits)

                loss_remaining = criterion(remaining_logits, remaining_labels)
                loss_unlearning = criterion(unlearning_logits, unlearning_labels)
                alpha = 1.0
                beta = 1.0
                loss = alpha*loss_remaining + beta*loss_unlearning
                loss.backward()

                opimizer.step()

                training_loss.append(loss.item() / labels.shape[0])
                loop.set_description(
                    f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
                )

            loop.close()

        return unlearning_model

    # # change classifier
    elif method == 'change':
        print('code not available')

    elif method == 'complementary':

        unlearning_model = retrain_classifier_complementary_method(original_model, unlearning_loader,
                                                                   device, lr, epochs)

        return unlearning_model

    else:
        print('unexpected method')
        exit()

    print('retrain classifier')


def randomly_initialize(original_model, raw_model, unlearning_loader, device, lr, epochs, aug_perturb):
    """

    :param original_model: model parametered with parameters of all patients
    :param raw_model: parameters random initilized
    :param unlearning_loader:
    :param device:
    :param lr:
    :param epochs:
    :param aug_perturb:
    :return:
    """

    # retrain classifier
    # # update classifier parameters
    unlearning_model = raw_model

    optimizer = optim.Adam(unlearning_model.parameters(), lr=lr)
    # criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    original_model.to(device)
    original_model.eval()
    unlearning_model.to(device)
    unlearning_model.train()

    # generate complemetary labels
    complementary_loader, complementary_labels_prior, num_classes = generate_complementary_labels(unlearning_loader)
    # unlearning with complementary labels
    complementary_methods = ['assumption free', 'non negative', 'forward']
    used_method = complementary_methods[0]

    for epoch in range(epochs):
        loop = tqdm(complementary_loader, ascii=True)
        training_loss = []
        for samples, labels_class, labels_domain, complementary_labels in loop:
            samples, labels_class, labels_domain, complementary_labels = \
                augmentation_perturbation(samples, labels_class, labels_domain, complementary_labels, aug_perturb)
            samples = samples.to(device)
            complementary_labels = complementary_labels.to(device)

            optimizer.zero_grad()
            dynamic_logits = unlearning_model(samples)
            static_logits = original_model(samples)

            remaining_logits_dynamic, unlearning_logits_dynamic, remaining_labels, unlearning_labels, \
                remaining_logits_static, unlearning_complementary_labels = \
                separate_prediction([labels_class, complementary_labels], labels_domain, device,
                                    static_logits, dynamic_logits)

            loss_unlearn, loss_vector = complementary_loss_choose(unlearning_logits_dynamic,
                                                                  unlearning_complementary_labels,
                                                                  num_classes, device, complementary_labels_prior,
                                                                  used_method)

            # loss_unlearn, loss_vector = complementary_loss_choose(dynamic_logits, complementary_labels,
            #                                                       num_classes, device, complementary_labels_prior,
            #                                                       used_method)

            # loss_remain = criterion_ce(remaining_logits_dynamic, remaining_labels)
            loss_remain = criterion_kl(torch.log(func.softmax(remaining_logits_dynamic, dim=-1)),
                                       func.softmax(remaining_logits_static, dim=-1))
            loss_T = loss_Tsallis(remaining_logits_dynamic)
            alpha = 1.0
            beta = 1.0
            gamma = 1.0

            if used_method == 'gradient ascent':
                if torch.min(loss_vector).item() < 0:
                    loss_vector_with_zeros = torch.cat((loss_vector.view(-1, 1),
                                                        torch.zeros(num_classes).view(-1, 1).to(device)), 1)
                    loss_vector_min, _ = torch.min(loss_vector_with_zeros, dim=1)
                    loss_unlearn = torch.sum(loss_vector_min)
                    loss = alpha * loss_unlearn + beta * loss_remain
                    loss.backward()

                    for group in optimizer.param_groups:
                        for p in group['params']:
                            p.grad = -p.grad
                else:
                    loss = alpha * loss_unlearn + beta * loss_remain
                    loss.backward()
            else:
                loss = alpha * loss_unlearn + beta * loss_remain + gamma * loss_T
                loss.backward()

            optimizer.step()

            training_loss.append(loss.item() / labels_class.shape[0])
            loop.set_description(
                f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
            )

        loop.close()

    return unlearning_model


def extra_states_method(unlearning_model, loader, device, lr, epochs, extra_states):

    feature_extrator = unlearning_model.featurer
    classifier = unlearning_model.classifier

    # change classifier (this process is different for different models)
    in_features = 2
    out_features = 2
    widen_classifier = nn.Linear(in_features, out_features + extra_states)
    widen_model = nn.Sequential(feature_extrator, widen_classifier)
    widen_model = widen_model.to(device)

    for name, params in classifier.named_parameters():
        if 'weight' in name:
            widen_classifier.state_dict()['weight'][0:out_features, :] = classifier.state_dict()[name][:, :]
        elif 'bias' in name:
            widen_classifier.state_dict()['bias'][0:out_features] = classifier.state_dict()[name][:]

    # unlearning
    optimizer = optim.Adam(widen_model.parameters(), lr=lr[1])
    criterion = nn.CrossEntropyLoss()
    widen_model.to(device)
    for epoch in range(epochs):
        loop = tqdm(loader, ascii=True)
        train_losses = []
        widen_model.train()
        for samples, labels in loop:
            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = widen_model(samples)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item() / labels.shape[0])
            loop.set_description(
                f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(train_losses) / len(train_losses)):.10f}"
            )

        loop.close()

    torch.save(widen_model[0].state_dict(), os.path.join('temp/featurer_para', 'widen_model_featurer_para.pth'))

    # restore model
    unlearning_model.featurer.load_state_dict(torch.load(os.path.join('temp/featurer_para',
                                                                      'widen_model_featurer_para.pth')))
    os.remove(os.path.join('temp/featurer_para', 'widen_model_featurer_para.pth'))

    # restore classifier
    for name, params in widen_model[1].named_parameters():
        # print(name)
        if 'weight' in name:
            unlearning_model.classifier.state_dict()['weight'][:, :] = \
                widen_model[1].state_dict()[name][0:out_features, :]
        elif 'bias' in name:
            unlearning_model.classifier.state_dict()['bias'][:] = widen_model[1].state_dict()[name][0:out_features]

    return unlearning_model


def reverse_labels_method(model, loader, device, lr, epochs):

    # unlearning
    # # unlearning under supervision of original model
    domain_labels = True
    if domain_labels:
        static_model = deepcopy(model)
        dynamic_model = deepcopy(model)

        optimizer = optim.Adam(dynamic_model.parameters(), lr=lr[1])
        # optimizer = optim.SGD(dynamic_model.parameters(), lr=lr[1], momentum=0.9)
        criterion_ce = nn.CrossEntropyLoss()
        # criterion_kl = nn.KLDivLoss(reduction='batchmean')
        # criterion_mse = nn.MSELoss(reduction='mean')
        dynamic_model.to(device)
        static_model.to(device)
        dynamic_model.train()
        for epoch in range(epochs):
            loop = tqdm(loader, ascii=True)
            train_losses = []
            # # # unlearning and restoring in one training phase
            for samples, labels, labels_domain in loop:
                samples = samples.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                static_logits = static_model(samples)
                dynamic_logits = dynamic_model(samples)

                remaining_logits_static, remaining_logits_dynamic, unlearning_logits_dynamic, \
                    remaining_labels, unlearning_labels = \
                    separate_prediction([labels], labels_domain, device, static_logits, dynamic_logits)

                # loss_static = criterion_kl(func.log_softmax(remaining_logits_dynamic, dim=-1),
                #                            func.softmax(remaining_logits_static, dim=-1))
                # loss_static = criterion_ce(remaining_logits_dynamic, remaining_logits_static.softmax(dim=1))

                loss_remain = criterion_ce(remaining_logits_dynamic, remaining_labels)
                loss_unlearn = criterion_ce(unlearning_logits_dynamic, unlearning_labels)

                alpha = 0  # recommond value = 5
                beta = 1
                # loss = alpha*loss_static + beta*loss_dynamic
                loss = alpha*loss_remain + beta*loss_unlearn
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item() / labels.shape[0])
                loop.set_description(
                    f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(train_losses) / len(train_losses)):.10f}"
                )

            loop.close()

            # # # seperate unlearning and restoring

        return dynamic_model

    # # unlearning under self-supervision
    else:
        unlearning_model = deepcopy(model)

        optimizer = optim.Adam(unlearning_model.parameters(), lr=lr[1])
        criterion = nn.CrossEntropyLoss()
        unlearning_model.to(device)
        for epoch in range(epochs):
            loop = tqdm(loader, ascii=True)
            train_losses = []
            unlearning_model.train()
            for samples, labels, _ in loop:
                samples = samples.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = unlearning_model(samples)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item() / labels.shape[0])
                loop.set_description(
                    f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(train_losses) / len(train_losses)):.10f}"
                )

            loop.close()

        return unlearning_model


def complementary_labels_method(model, loader, device, lr, epochs):
    """
        forget specific patient's data
        :param model: model with trained parameters
        :param loader: tensor in dataloader, [0]: samples; [1]: class labels; [2]: domain labels
        :param device: gpu if 'cuda' else cpu
        :param lr: float in list, [0]: repairing learning rate; [1] unlearning learning rate
        :param epochs:
        :return: unlearned model
        """
    # generate complemetary labels
    complementary_loader, complementary_labels_prior, num_classes = generate_complementary_labels(loader)

    # unlearning with complementary labels
    complementary_methods = ['assumption free', 'non negative', 'gradient ascent', 'forward']
    used_method = complementary_methods[0]
    unlearning_model = deepcopy(model)
    optimizer = optim.Adam(unlearning_model.parameters(), lr=lr[1])
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')

    for epoch in range(epochs):
        loop = tqdm(complementary_loader, ascii=True)
        train_losses = []
        unlearning_model.to(device)
        unlearning_model.train()
        model.eval()
        for samples, labels_class, labels_domain, complementary_labels in loop:
            samples = samples.to(device)
            complementary_labels = complementary_labels.to(device)

            optimizer.zero_grad()
            dynamic_logits = unlearning_model(samples)
            static_logits = model(samples)

            remaining_logits_dynamic, unlearning_logits_dynamic, remaining_labels, unlearning_labels, \
                remaining_logits_static, unlearning_complementary_labels = \
                separate_prediction([labels_class, complementary_labels], labels_domain, device,
                                    static_logits, dynamic_logits)

            loss_unlearn, loss_vector = complementary_loss_choose(unlearning_logits_dynamic,
                                                                  unlearning_complementary_labels,
                                                                  num_classes, device, complementary_labels_prior,
                                                                  used_method)
            # loss_unlearn, loss_vector = complementary_loss_choose(dynamic_logits, complementary_labels,
            #                                                       num_classes, device, complementary_labels_prior,
            #                                                       used_method)

            # loss_remain = criterion_ce(remaining_logits, remaining_labels)
            loss_remain = criterion_kl(torch.log(func.softmax(remaining_logits_dynamic, dim=-1)),
                                       func.softmax(remaining_logits_static, dim=-1))
            loss_T = loss_Tsallis(remaining_logits_dynamic)
            alpha = 1.0
            beta = 5.0
            gamma = 1.0

            if used_method == 'gradient ascent':
                if torch.min(loss_vector).item() < 0:
                    loss_vector_with_zeros = torch.cat((loss_vector.view(-1, 1),
                                                        torch.zeros(num_classes).view(-1, 1).to(device)), 1)
                    loss_vector_min, _ = torch.min(loss_vector_with_zeros, dim=1)
                    loss_unlearn = torch.sum(loss_vector_min)
                    loss = alpha*loss_unlearn + beta*loss_remain
                    loss.backward()

                    for group in optimizer.param_groups:
                        for p in group['params']:
                            p.grad = -p.grad
                else:
                    loss = alpha*loss_unlearn + beta*loss_remain
                    loss.backward()
            else:
                loss = alpha * loss_unlearn + beta * loss_remain + gamma * loss_T
                loss.backward()

            optimizer.step()

            train_losses.append(loss.item() / complementary_labels.shape[0])
            loop.set_description(
                f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(train_losses) / len(train_losses)):.10f}"
            )

        loop.close()

    return unlearning_model


def retrain_featurer_complementary_method(model, loader, device, lr, epochs, aug_pertur):
    """

    :param model:
    :param loader:
    :param device:
    :param lr:
    :param epochs:
    :param aug_pertur:
    :return:
    """

    # retrain classifier
    # # update classifier parameters
    unlearning_model = deepcopy(model)

    optimizer = optim.Adam(unlearning_model.featurer.parameters(), lr=lr)
    # criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    model.to(device)
    model.eval()
    unlearning_model.to(device)
    unlearning_model.train()
    for param in unlearning_model.classifier.parameters():
        param.required_grad = False

    # generate complemetary labels
    complementary_loader, complementary_labels_prior, num_classes = generate_complementary_labels(loader)
    # unlearning with complementary labels
    complementary_methods = ['assumption free', 'non negative', 'forward']
    used_method = complementary_methods[0]

    for epoch in range(epochs):
        loop = tqdm(complementary_loader, ascii=True)
        training_loss = []
        for samples, labels_class, labels_domain, complementary_labels in loop:
            samples, labels_class, labels_domain, complementary_labels = \
                augmentation_perturbation(samples, labels_class, labels_domain, complementary_labels, aug_pertur)
            samples = samples.to(device)
            complementary_labels = complementary_labels.to(device)

            optimizer.zero_grad()
            dynamic_logits = unlearning_model(samples)
            static_logits = model(samples)

            remaining_logits_dynamic, unlearning_logits_dynamic, remaining_labels, unlearning_labels, \
                remaining_logits_static, unlearning_complementary_labels = \
                separate_prediction([labels_class, complementary_labels], labels_domain, device,
                                    static_logits, dynamic_logits)

            loss_unlearn, loss_vector = complementary_loss_choose(unlearning_logits_dynamic,
                                                                  unlearning_complementary_labels,
                                                                  num_classes, device, complementary_labels_prior,
                                                                  used_method)

            # loss_unlearn, loss_vector = complementary_loss_choose(dynamic_logits, complementary_labels,
            #                                                       num_classes, device, complementary_labels_prior,
            #                                                       used_method)

            # loss_remain = criterion_ce(remaining_logits_dynamic, remaining_labels)
            loss_remain = criterion_kl(torch.log(func.softmax(remaining_logits_dynamic, dim=-1)),
                                       func.softmax(remaining_logits_static, dim=-1))
            loss_T = loss_Tsallis(remaining_logits_dynamic)
            alpha = 5.0
            beta = 1.0
            gamma = 1.0

            if used_method == 'gradient ascent':
                if torch.min(loss_vector).item() < 0:
                    loss_vector_with_zeros = torch.cat((loss_vector.view(-1, 1),
                                                        torch.zeros(num_classes).view(-1, 1).to(device)), 1)
                    loss_vector_min, _ = torch.min(loss_vector_with_zeros, dim=1)
                    loss_unlearn = torch.sum(loss_vector_min)
                    loss = alpha * loss_unlearn + beta * loss_remain
                    loss.backward()

                    for group in optimizer.param_groups:
                        for p in group['params']:
                            p.grad = -p.grad
                else:
                    loss = alpha * loss_unlearn + beta * loss_remain
                    loss.backward()
            else:
                loss = alpha * loss_unlearn + beta * loss_remain + gamma * loss_T
                loss.backward()

            optimizer.step()

            training_loss.append(loss.item() / labels_class.shape[0])
            loop.set_description(
                f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
            )

        loop.close()

    # # work with original parameters
    # source_parameters = model.to(device).state_dict()
    # unlearned_parameters = unlearning_model.state_dict()
    # for key in source_parameters:
    #     if 'featurer' in key:
    #         if 'weight' in key:
    #             unlearned_parameters[key] += 0.1 * source_parameters[key]
    #         elif 'bias' in key:
    #             unlearned_parameters[key] += 0.1 * source_parameters[key]
    #         else:
    #             pass
    #     elif 'classifier' in key:
    #         pass
    #     else:
    #         pass
    # unlearning_model.load_state_dict(unlearned_parameters)

    return unlearning_model


def retrain_classifier_complementary_method(model, loader, device, lr, epochs):
    """

    :param model:
    :param loader:
    :param device:
    :param lr:
    :param epochs:
    :return:
    """

    # retrain classifier
    # # update classifier parameters
    unlearning_model = deepcopy(model)

    optimizer = optim.Adam(unlearning_model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    unlearning_model.to(device)
    unlearning_model.train()
    for param in unlearning_model.featurer.parameters():
        param.required_grad = False

    # generate complemetary labels
    complementary_loader, complementary_labels_prior, num_classes = generate_complementary_labels(loader)
    # unlearning with complementary labels
    complementary_methods = ['assumption free', 'non negative', 'forward']
    used_method = complementary_methods[0]

    for epoch in range(epochs):
        loop = tqdm(complementary_loader, ascii=True)
        training_loss = []
        for samples, labels_class, labels_domain, complementary_labels in loop:
            samples = samples.to(device)
            complementary_labels = complementary_labels.to(device)

            optimizer.zero_grad()
            dynamic_logits = unlearning_model(samples)

            remaining_logits, unlearning_logits, remaining_labels, unlearning_labels, \
                unlearning_complementary_labels = separate_prediction([labels_class, complementary_labels],
                                                                      labels_domain, device, dynamic_logits)

            loss_unlearn, loss_vector = complementary_loss_choose(unlearning_logits, unlearning_complementary_labels,
                                                                  num_classes, device, complementary_labels_prior,
                                                                  used_method)

            # loss_unlearn, loss_vector = complementary_loss_choose(dynamic_logits, complementary_labels,
            #                                                       num_classes, device, complementary_labels_prior,
            #                                                       used_method)

            loss_remain = criterion(remaining_logits, remaining_labels)
            alpha = 1.0
            beta = 1.0

            if used_method == 'gradient ascent':
                if torch.min(loss_vector).item() < 0:
                    loss_vector_with_zeros = torch.cat((loss_vector.view(-1, 1),
                                                        torch.zeros(num_classes).view(-1, 1).to(device)), 1)
                    loss_vector_min, _ = torch.min(loss_vector_with_zeros, dim=1)
                    loss_unlearn = torch.sum(loss_vector_min)
                    loss = alpha * loss_unlearn + beta * loss_remain
                    loss.backward()

                    for group in optimizer.param_groups:
                        for p in group['params']:
                            p.grad = -p.grad
                else:
                    loss = alpha * loss_unlearn + beta * loss_remain
                    loss.backward()
            else:
                loss = alpha * loss_unlearn + beta * loss_remain
                loss.backward()

            optimizer.step()

            training_loss.append(loss.item() / labels_class.shape[0])
            loop.set_description(
                f"Epoch: {epoch + 1}/{epochs} | Epoch loss: {(sum(training_loss) / len(training_loss)):.10f}"
            )

        loop.close()

    return unlearning_model
