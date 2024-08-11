import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from PCT_net import PCT


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
        labels_f_d = labels_f_d.to(device)
        samples_f_r = samples_f_r.to(device)
        labels_f_r = labels_f_r.to(device)

        optimizer.zero_grad()

        logits_f_r = model(samples_f_r)
        loss_f_r = criterion(logits_f_r, labels_f_r)
        logits_f_d = model(samples_f_d)
        loss_f_d = criterion(logits_f_d, labels_f_d)

        loss = loss_f_d - loss_f_r

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


if __name__ == '__main__':
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

    print('influence function functions')