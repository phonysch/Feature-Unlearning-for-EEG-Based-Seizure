import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from PCT_net import PCT
import random
from copy import deepcopy
import torch.nn.functional as func


def get_gradients(model, loader_forgetting_raw, loader_forgetting_disturb, lr, device):
    """

    :param model: model with weights
    :param loader_forgetting_raw: dataloader; forgetting patients' data without disturbing
    :param loader_forgetting_disturb: dataloader; forgetting patients' data with disturbed samples or labels.
    :param lr: float in list; [0] automatically learning rate, [1] manually learning rate
    :param device: cpu or cuda
    :return: gradients in list
    """

    optimizer = optim.Adam(model.parameters(), lr=lr[0])
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    loop = tqdm(loader_forgetting_raw, ascii=True)
    loader_disturb = iter(loader_forgetting_disturb)
    model.train()

    grads1 = []
    for samples_f_r, labels_f_r in loop:
        samples_f_d, labels_f_d = next(loader_disturb)
        samples_f_d = samples_f_d.to(device)
        # labels_f_d = labels_f_d.to(device)
        samples_f_r = samples_f_r.to(device)
        # labels_f_r = labels_f_r.to(device)

        optimizer.zero_grad()

        logits_f_r = model(samples_f_r)
        logits_f_d = model(samples_f_d)

        # loss_f_r = criterion(logits_f_r, labels_f_r)
        # loss_f_d = criterion(logits_f_d, labels_f_d)
        # loss = loss_f_d - loss_f_r

        loss = criterion(logits_f_r, logits_f_d.softmax(dim=1))

        loss.backward()  #

        grads1.append(get_model_gradient(model))

        loop.set_description(
            f"processing"
        )

    loop.close()

    grads_sum1 = list(zip(*grads1))

    for i in range(len(grads_sum1)):
        sum_ = 0
        for text in grads_sum1[i]:
            sum_ += text
        grads_sum1[i] = sum_

    return grads_sum1


def get_model_gradient(model):
    """

    :param model: model with gradients
    :return: gradients
    """

    grads = []

    for name, params in model.named_parameters():
        grad = params.grad
        if grad is not None:
            grads.append(grad)

    return grads


def update_model_parameters(model, gradients, update_factor):

    parameters = model.state_dict()

    # Parameters that do not need to be updated
    name_list = ['running_mean', 'running_var', 'num_batches_tracked']

    gradients_iter = iter(gradients)
    for name in parameters:
        if name.split('.')[-1] not in name_list:
            parameters[name] -= update_factor * next(gradients_iter)

    model.load_state_dict(parameters)

    return model


def repairing_train(model, loader, device, repairing_lr=0.00001, repairing_epoch=1):

    optimizer = optim.Adam(model.parameters(), lr=repairing_lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(repairing_epoch):
        loop = tqdm(loader, ascii=True)
        train_losses = []
        model.train()
        for samples, labels in loop:

            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(samples)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item() / labels.shape[0])
            loop.set_description(
                f"Epoch: {epoch + 1}/{repairing_epoch} | Epoch loss: {(sum(train_losses) / len(train_losses)):.10f}"
            )

        loop.close()

    return model


def separate_prediction(labels_class, labels_domain, device, *predictions):
    """
    seperate logits of remaining patients and unlearning patient
    :param labels_class: tensor in list, if len(labels_class) == 1:
                                            [0]: '0' for interictal samples, '1' for preictal samples
                                         elif len(labels_class) == 2:
                                            [0]: '0' for interictal samples, '1' for preictal samples
                                            [1]: complementary labels
    :param labels_domain: tensor, '0' for samples from remaining patients, '1' for samples from unlearning patient.
    :param device:
    :param predictions: tensor, if len(predictions) == 2: [0]: outputs of static model; [1]: outputs of dynamic model
                                elif len(predictions) == 1: logic outputs
    :return:
    """
    labels_domain_numpy = deepcopy(labels_domain).cpu().numpy()
    num_u = len(labels_domain_numpy[labels_domain_numpy == 1])
    num_r = len(labels_domain_numpy[labels_domain_numpy == 0])
    unlearning_labels, remaining_labels = [], []

    if len(predictions) == 1:
        if len(labels_class) == 1:
            remaining_logits, unlearning_logits = [], []
            for ind, label in enumerate(labels_domain_numpy):
                if int(label) == 0:
                    remaining_logits.append(predictions[0][ind])
                    remaining_labels.append(labels_class[0][ind])
                elif int(label) == 1:
                    unlearning_logits.append(predictions[0][ind])
                    unlearning_labels.append(labels_class[0][ind])
                else:
                    print('unexcepted labels')
                    exit()

            try:
                remaining_logits = torch.cat(remaining_logits).view(num_r, -1)
                remaining_labels = torch.tensor(remaining_labels, dtype=torch.int64, device=device)
            except NotImplementedError:
                remaining_logits = torch.zeros((1, 2), dtype=torch.float32, device=device)
                remaining_labels = torch.zeros((1,), dtype=torch.int64, device=device)
                print('lack remaining samples')
                pass

            try:
                unlearning_logits = torch.cat(unlearning_logits).view(num_u, -1)
                unlearning_labels = torch.tensor(unlearning_labels, dtype=torch.int64, device=device)
            except NotImplementedError:
                unlearning_logits = torch.zeros((1, 2), dtype=torch.float32, device=device)
                unlearning_labels = torch.zeros((1,), dtype=torch.int64, device=device)
                print('lack unlearning samples')
                pass

            return remaining_logits, unlearning_logits, remaining_labels, unlearning_labels

        elif len(labels_class) == 2:
            remaining_logits, unlearning_logits, unlearning_complementary_labels = [], [], []
            for ind, label in enumerate(labels_domain_numpy):
                if int(label) == 0:
                    remaining_logits.append(predictions[0][ind])
                    remaining_labels.append(labels_class[0][ind])
                elif int(label) == 1:
                    unlearning_logits.append(predictions[0][ind])
                    unlearning_labels.append(labels_class[0][ind])
                    unlearning_complementary_labels.append(labels_class[1][ind])
                else:
                    print('unexcepted labels')
                    exit()

            try:
                remaining_logits = torch.cat(remaining_logits).view(num_r, -1)
                remaining_labels = torch.tensor(remaining_labels, dtype=torch.int64, device=device)
            except NotImplementedError:
                remaining_logits = torch.zeros((1, 2), dtype=torch.float32, device=device)
                remaining_labels = torch.zeros((1,), dtype=torch.int64, device=device)
                print('lack remaining samples')
                pass

            try:
                unlearning_logits = torch.cat(unlearning_logits).view(num_u, -1)
                unlearning_labels = torch.tensor(unlearning_labels, dtype=torch.int64, device=device)
                unlearning_complementary_labels = torch.tensor(unlearning_complementary_labels, dtype=torch.int64,
                                                               device=device)
            except NotImplementedError:
                unlearning_logits = torch.zeros((1, 2), dtype=torch.float32, device=device)
                unlearning_labels = torch.zeros((1,), dtype=torch.int64, device=device)
                unlearning_complementary_labels = torch.zeros((1,), dtype=torch.int64, device=device)
                # print('lack unlearning samples')
                pass

            return remaining_logits, unlearning_logits, remaining_labels, unlearning_labels,\
                unlearning_complementary_labels

        else:
            print('not coded yet')

    elif len(predictions) == 2:
        if len(labels_class) == 1:
            remaining_logits_static, remaining_logits_dynamic, unlearning_logits_dynamic,  = [], [], []
            for ind, label in enumerate(labels_domain_numpy):
                if int(label) == 0:
                    remaining_logits_static.append(predictions[0][ind])
                    remaining_logits_dynamic.append(predictions[1][ind])
                    remaining_labels.append(labels_class[0][ind])
                elif int(label) == 1:
                    unlearning_logits_dynamic.append(predictions[1][ind])
                    unlearning_labels.append(labels_class[0][ind])

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

            return remaining_logits_dynamic, unlearning_logits_dynamic, \
                remaining_labels, unlearning_labels, remaining_logits_static

        elif len(labels_class) == 2:
            remaining_logits_static, remaining_logits_dynamic, unlearning_logits_dynamic, \
                unlearning_complementary_labels = [], [], [], []
            for ind, label in enumerate(labels_domain_numpy):
                if int(label) == 0:
                    remaining_logits_static.append(predictions[0][ind])
                    remaining_logits_dynamic.append(predictions[1][ind])
                    remaining_labels.append(labels_class[0][ind])
                elif int(label) == 1:
                    unlearning_logits_dynamic.append(predictions[1][ind])
                    unlearning_labels.append(labels_class[0][ind])
                    unlearning_complementary_labels.append(labels_class[1][ind])
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
                unlearning_complementary_labels = torch.tensor(unlearning_complementary_labels, dtype=torch.int64,
                                                               device=device)
            except NotImplementedError:
                unlearning_logits_dynamic = torch.zeros((1, 2), dtype=torch.float32, device=device)
                unlearning_labels = torch.zeros((1,), dtype=torch.int64, device=device)
                unlearning_complementary_labels = torch.zeros((1,), dtype=torch.int64, device=device)
                # print('lack unlearning samples')
                pass

            return remaining_logits_dynamic, unlearning_logits_dynamic, \
                remaining_labels, unlearning_labels, remaining_logits_static, unlearning_complementary_labels

    else:
        print("arguments 'predictions' error")
        exit()


# def separate_samples():


def generate_complementary_labels(dataloader):
    # get all data and corresponding labels
    all_samples, all_labels, all_labels_domain = [], [], []
    for samples, labels, labels_domain in dataloader:
        all_samples.append(samples)
        all_labels.append(labels)
        all_labels_domain.append(labels_domain)
    batch_size = len(all_labels[0])
    all_samples = torch.cat(all_samples)
    all_labels = torch.cat(all_labels)
    all_labels_domain = torch.cat(all_labels_domain)

    # generate complementary labels
    classes = int((torch.max(all_labels)+1).cpu().numpy())  # expand 'n' shadow classes by adding '(1+n)' instead of '1'
    candidates = np.repeat(np.arange(classes).reshape(1, classes), len(all_labels), 0)
    mask = np.ones((len(all_labels), classes), dtype=bool)
    mask[range(len(all_labels)), all_labels.numpy()] = False
    candidates = candidates[mask].reshape(len(all_labels), classes-1)
    complementary_idx = np.random.randint(0, classes-1, len(all_labels))
    complementary_labels = candidates[np.arange(len(all_labels)), complementary_idx]

    # # get all complementary labels prior
    # prior = np.bincount(complementary_labels) / len(complementary_labels)
    # get unlearning complementary labels prior
    unlearning_complementary_labels = []
    for index, labels in enumerate(all_labels_domain):
        if labels == 1:
            unlearning_complementary_labels.append(all_labels[index])
    unlearning_complementary_labels = torch.tensor(unlearning_complementary_labels)
    prior = np.bincount(unlearning_complementary_labels) / len(unlearning_complementary_labels)

    complementary_labels = torch.from_numpy(complementary_labels)
    complementary_set = data.TensorDataset(all_samples, all_labels, all_labels_domain, complementary_labels)
    complementary_loader = data.DataLoader(
        dataset=complementary_set,
        batch_size=batch_size,
        shuffle=False
    )

    return complementary_loader, prior, classes


def complementary_loss_choose(logits, labels, num_classes, device, complementary_prior, complementary_method):
    if complementary_method == 'assumption free':

        loss, loss_vector = assumption_free_loss(logits, num_classes, labels, device, complementary_prior)

        return loss, loss_vector

    elif complementary_method == 'non negative':

        loss, loss_vector = non_negative_loss(logits, num_classes, labels, device, complementary_prior, 0)

        return loss, loss_vector

    elif complementary_method == 'gradient ascent':

        loss, loss_vector = gradient_ascent_loss(logits, num_classes, labels, device, complementary_prior)

        return loss, loss_vector

    elif complementary_method == 'forward':

        loss, loss_vector = forward_loss(logits, num_classes, labels, device)

        return loss, loss_vector

    else:
        print('unexpected complementary method')


def non_negative_loss(logits, num_classes, labels, device, complementary_prior, beta):
    complementary_prior = torch.from_numpy(complementary_prior.astype(np.float32)).to(device)
    logits = -func.softmax(logits, dim=1)
    loss_vector = torch.zeros(num_classes).to(device)
    # # criterion = torch.nn.CrossEntropyLoss()

    for cla in range(num_classes):
        idx = labels == cla
        if torch.sum(idx).item() > 0:
            idx_logits = idx.view(-1, 1).repeat(1, num_classes)
            logits_masked = torch.masked_select(logits, idx_logits).view(-1, num_classes)
            # # labels_masked = torch.masked_select(labels, idx).view(-1,)
            # loss_vector[cla] = -(num_classes-1) * complementary_prior[cla] * torch.mean(logits_masked, dim=0)[cla]
            loss_vector[cla] = -num_classes * complementary_prior[cla] * torch.mean(logits_masked, dim=0)[cla]
            loss_vector = loss_vector + torch.mul(complementary_prior, torch.mean(logits_masked, dim=0))
            # # loss_vector[cla] = -num_classes * complementary_prior[cla] * criterion(logits_masked, labels_masked)
            # # loss_vector = loss_vector + complementary_prior[cla] * criterion(logits_masked, labels_masked)

    count = np.bincount(labels.cpu().numpy())
    while len(count) < num_classes:
        count = np.append(count, 0)

    loss_vector_beta = torch.cat((loss_vector.view(-1, 1), torch.zeros(num_classes).view(-1, 1).to(device)-beta), 1)
    loss_vector_max, _ = torch.max(loss_vector_beta, dim=1)
    loss = torch.sum(loss_vector_max)

    # # loss = torch.sum(loss_vector) / len(loss_vector)

    return loss, None


def gradient_ascent_loss(logits, num_classes, labels, device, complementary_prior):
    # equivalent with non_negative_loss when the max operator is negtive inf
    beta = np.inf
    loss, loss_vector = non_negative_loss(logits, num_classes, labels, device, complementary_prior, beta)

    return loss, loss_vector


def assumption_free_loss(logits, num_classes, labels, device, complementary_prior):
    # equivalent with non_negative_loss when the max operator is negtive inf
    beta = np.inf
    loss, loss_vector = non_negative_loss(logits, num_classes, labels, device, complementary_prior, beta)

    return loss, loss_vector


def forward_loss(logits, num_classes, labels, device):
    factor = torch.ones(num_classes, num_classes, device=device) * 1/(num_classes-1)

    for n in range(num_classes):
        factor[n, n] = 0

    complementary_logits = torch.mm(func.softmax(logits, dim=1), factor)

    return func.nll_loss(complementary_logits.log(), labels), None


def loss_Tsallis(logits):

    # probability rescaling
    rescal_logits = []
    t = 2

    for index1, text1 in enumerate(logits):
        sum_exp = sum(torch.exp(text1/t))
        for text2 in text1:
            exp_logit = torch.exp(text2/t)
            rescal_logits.append((exp_logit/sum_exp).view(1,))
    rescal_logits = torch.cat(rescal_logits).view(logits.shape[0], logits.shape[1])

    # sample reweighting
    n = len(logits)
    w = []
    denominator = 0
    for i in range(len(rescal_logits)):
        denominator += 1 + torch.exp(sum(rescal_logits[i] * torch.log(rescal_logits[i])))
    for i in range(len(rescal_logits)):
        numerator = n * (1 + torch.exp(sum(rescal_logits[i] * torch.log(rescal_logits[i]))))
        w.append(numerator/denominator)

    # # category normalization
    # r = []
    # for i in range(rescal_logits.shape[1]):
    #     # a = rescal_logits[0:-1, i]
    #     r.append(sum(rescal_logits[0:-1, i]))
    #
    # a = 2
    # num_classes = logits.shape[1]
    # element_logits = []
    # try:
    #     for i in range(len(rescal_logits)):
    #         temp_logits = 0.0
    #         for j in range(rescal_logits.shape[1]):
    #             temp_logits += (rescal_logits[i][j] ** a / torch.exp(r[j]))
    #             # when lack remaining samples, 'r' will be 0
    #             # temp_logits += (rescal_logits[i][j] ** a)
    #         element_logits.append(w[i] * temp_logits)
    #
    #     sum_logits = sum(element_logits)
    #     loss = - sum_logits / (n * num_classes * (a - 1))
    # except TypeError:
    #     loss = 0

    # without category normalization
    loss = 0
    power = 2
    for ind, logit in enumerate(rescal_logits):
        loss += (1 - w[ind] * sum(torch.pow(logit, power)))
    loss = loss / (len(rescal_logits) * (power-1))

    return loss


def augmentation_perturbation(samples, labels_class, labels_domain, complementary_labels, aug_perturb):
    """

    :param samples:
    :param labels_domain: tensor, '0' for samples from remaining patients, '1' for samples from unlearning patient.
    :param labels_class:
    :param complementary_labels:
    :param aug_perturb:
    :return:
    """
    labels_domain_numpy = deepcopy(labels_domain).cpu().numpy()

    if aug_perturb == 0:
        augment_samples, augment_labels_class, augment_labels_domain, augment_complementary_labels = [], [], [], []
        for ind, label in enumerate(labels_domain_numpy):
            if int(label) == 0:
                augment_samples.append(samples[ind])
                augment_labels_class.append(labels_class[ind])
                augment_labels_domain.append(labels_domain[ind])
                augment_complementary_labels.append(complementary_labels[ind])
            elif int(label) == 1:
                augment_samples.append(samples[ind])
                augment_labels_class.append(labels_class[ind])
                augment_labels_domain.append(labels_domain[ind])
                augment_complementary_labels.append(complementary_labels[ind])
            else:
                print('unexcepted labels')
                exit()

        augment_samples = torch.cat(augment_samples)
        augment_samples = augment_samples.numpy()
        for sample in augment_samples:
            channel_changing_index = random.sample(range(0, sample.shape[1]), int(sample.shape[1]/2))
            factor = np.random.rand(int(sample.shape[1]/2),)
            for i in range(int(sample.shape[1]/2)):
                sample[:, channel_changing_index[i]] = sample[:, channel_changing_index[i]] * factor[i]
        augment_samples = augment_samples[:, np.newaxis, :, :]
        augment_samples = torch.from_numpy(augment_samples)

        augment_labels_class = torch.tensor(augment_labels_class)
        augment_labels_domain = torch.tensor(augment_labels_domain)
        augment_complementary_labels = torch.tensor(augment_complementary_labels)

        mixed_samples = torch.cat((samples, augment_samples))
        mixed_labels_class = torch.cat((labels_class, augment_labels_class))
        mixed_labels_domain = torch.cat((labels_domain, augment_labels_domain))
        mixed_complementary_labels = torch.cat((complementary_labels, augment_complementary_labels))

        return mixed_samples, mixed_labels_class, mixed_labels_domain, mixed_complementary_labels

    elif aug_perturb == 1:
        mask_or_add = False  # True for mask channel data, False for add random noise
        samples = samples.numpy()
        for ind, label in enumerate(labels_domain_numpy):
            if int(label) == 0:
                continue
            elif int(label) == 1:
                if mask_or_add:
                    # this method easily generate nan loss in some case after some eopch
                    channel_changing_index = random.sample(range(0, samples.shape[3]), int(samples.shape[3] / 4))
                    for index in channel_changing_index:
                        samples[ind][0, :, index] = samples[ind][0, :, index] * 0.01  # multiple 0 will generte nan loss
                else:
                    noise = np.random.rand(samples.shape[1], samples.shape[2], samples.shape[3]).astype(np.float32)
                    for index in range(len(samples)):
                        samples[ind] = samples[ind] + noise
            else:
                print('unexcepted labels')
                exit()
        samples = torch.from_numpy(samples.astype(np.float32))

        return samples, labels_class, labels_domain, complementary_labels

    else:

        return samples, labels_class, labels_domain, complementary_labels


if __name__ == '__main__':
    # debug 'get_gradients' and 'updata_model_parameters'
    model_ = PCT()

    raw_data = np.random.rand(1024, 1, 1024, 22).astype(np.float32)
    raw_labels = np.concatenate((np.ones(512, dtype=np.int64), np.zeros(512, dtype=np.int64)))
    raw_data, raw_labels = torch.from_numpy(raw_data), torch.from_numpy(raw_labels)
    raw_set = data.TensorDataset(raw_data, raw_labels)
    raw_loader = data.DataLoader(
        dataset=raw_set,
        batch_size=32,
        shuffle=True,
    )

    gradients_ = get_gradients(model_, raw_loader, raw_loader, [0.0001, 0.00001], 'cuda')

    unlearned_model_ = update_model_parameters(model_, gradients_, 0.00001)

    # # debug 'generate complementary labels'
    # s = np.random.randn(1024, 1, 22, 1024).astype(np.float32)
    # la = np.random.randint(0, 2, 1024).astype(np.int64)
    # s = torch.from_numpy(s)
    # la = torch.from_numpy(la)
    # set_ = data.TensorDataset(s, la)
    # loader_ = data.DataLoader(dataset=set_, batch_size=64, shuffle=True)
    #
    # generate_complementary_labels(loader_)

    # # debug 'non_negative_loss'
    # device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # s = np.random.randn(32, 2).astype(np.float32)
    # la = np.random.randint(0, 2, 32).astype(np.int64)
    # s = torch.from_numpy(s).to(device_)
    # la = torch.from_numpy(la).to(device_)
    # num = 2
    # prior_ = np.array([0.5, 0.5])
    # beta_ = 0
    #
    # non_negative_loss(s, num, la, device_, prior_, beta_)

    # test_logits = torch.ones(16, 2, dtype=torch.float32)*0.5
    # test_loss = loss_Tsallis(test_logits)

    print('unlearning scheme functions')
