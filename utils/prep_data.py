import torch
import numpy as np
import torch.utils.data as data


# def train_val_test_split(ictal_X, ictal_y, interictal_X, interictal_y, val_ratio, test_ratio):
#
#     num_sz = len(ictal_y)
#     num_sz_test = int(test_ratio * num_sz)  # int() ：向下取整
#     print('Total %d seizures. Last %d is used for testing.' % (num_sz, num_sz_test))
#
#     if isinstance(interictal_y, list):
#         interictal_X = np.concatenate(interictal_X, axis=0)
#         interictal_y = np.concatenate(interictal_y, axis=0)
#     interictal_fold_len = int(round(1.0 * interictal_y.shape[0] / num_sz))  # https://www.runoob.com/python/func-number-round.html
#     print('interictal_fold_len', interictal_fold_len)
#
#
#     X_test_ictal = np.concatenate(ictal_X[-num_sz_test:])  # - 表示从右边开始数 (CHBMIT---cbh14)等价于 [4 * num_sz_test:]
#     y_test_ictal = np.concatenate(ictal_y[-num_sz_test:])
#
#     X_test_interictal = interictal_X[-num_sz_test*interictal_fold_len:]
#     y_test_interictal = interictal_y[-num_sz_test*interictal_fold_len:]
#
#     X_train_ictal = np.concatenate(ictal_X[:-num_sz_test], axis=0)
#     y_train_ictal = np.concatenate(ictal_y[:-num_sz_test], axis=0)
#
#     X_train_interictal = interictal_X[:-num_sz_test*interictal_fold_len]
#     y_train_interictal = interictal_y[:-num_sz_test*interictal_fold_len]
#
#     print(y_train_ictal.shape, y_train_interictal.shape)
#
#     '''
#     Downsampling interictal training set so that the 2 classes
#     are balanced
#     '''
#     down_spl = int(np.floor(y_train_interictal.shape[0]/y_train_ictal.shape[0]))
#     if down_spl > 1:
#         X_train_interictal = X_train_interictal[::down_spl]
#         y_train_interictal = y_train_interictal[::down_spl]
#     elif down_spl == 1:
#         X_train_interictal = X_train_interictal[:X_train_ictal.shape[0]]
#         y_train_interictal = y_train_interictal[:y_train_ictal.shape[0]]
#
#     print('Balancing:', y_train_ictal.shape, y_train_interictal.shape)
#
#     #X_train_ictal = shuffle(X_train_ictal,random_state=0)
#     #X_train_interictal = shuffle(X_train_interictal,random_state=0)
#     X_train = np.concatenate((X_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))], X_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]), axis=0)
#     y_train = np.concatenate((y_train_ictal[:int(X_train_ictal.shape[0]*(1-val_ratio))], y_train_interictal[:int(X_train_interictal.shape[0]*(1-val_ratio))]), axis=0)
#
#     X_val = np.concatenate((X_train_ictal[-int(X_train_ictal.shape[0]*val_ratio):], X_train_interictal[int(-X_train_interictal.shape[0]*val_ratio):]), axis=0)
#     y_val = np.concatenate((y_train_ictal[-int(X_train_ictal.shape[0]*val_ratio):], y_train_interictal[int(-X_train_interictal.shape[0]*val_ratio):]), axis=0)
#
#     nb_val = X_val.shape[0] - X_val.shape[0] % 4  # 什么意思
#     X_val = X_val[:nb_val]
#     y_val = y_val[:nb_val]
#
#     # let overlapped ictal samples have same labels with non-overlapped samples
#     y_train[y_train == 2] = 1
#     y_val[y_val == 2] = 1
#
#     X_test = np.concatenate((X_test_ictal, X_test_interictal), axis=0)
#     y_test = np.concatenate((y_test_ictal, y_test_interictal), axis=0)
#
#     # remove overlapped ictal samples in test-set
#     X_test = X_test[y_test != 2]
#     y_test = y_test[y_test != 2]
#
#     print('X_train, X_val, X_test', X_train.shape, X_val.shape, X_test.shape)
#     return X_train, y_train, X_val, y_val, X_test, y_test
#
#
# def train_val_test_split_sch(ictal_sample, ictal_label, interictal_sample, interictal_label, val_ratio, test_ratio):
#
#     num_sz = len(ictal_label)
#     num_sz_test = int(test_ratio * num_sz)  # int() ：向下取整
#     if num_sz_test == 0:
#         num_sz_test = 1
#     print('Total %d seizures. Last %d is used for testing.' % (num_sz, num_sz_test))
#
#     sample_train_ictal = ictal_sample[:-num_sz_test]
#     label_train_ictal = ictal_label[:-num_sz_test]
#     sample_train_interictal = interictal_sample[:-num_sz_test]
#     label_train_interictal = interictal_label[:-num_sz_test]
#
#     sample_test_ictal = ictal_sample[-num_sz_test:]
#     label_test_ictal = ictal_label[-num_sz_test:]
#     sample_test_interictal = interictal_sample[-num_sz_test:]
#     label_test_interictal = interictal_label[-num_sz_test:]
#
#     # 设置测试集
#     sample_test_ictal = sample_test_ictal[0]
#     label_test_ictal = label_test_ictal[0]
#     sample_test_interictal = sample_test_interictal[0]
#     label_test_interictal = label_test_interictal[0]
#
#     # 删除label=2的label和sample
#     temp = 0
#     for i in range(len(label_test_ictal)):
#         if label_test_ictal[temp] == 2:
#             label_test_ictal.pop(temp)
#             sample_test_ictal.pop(temp)
#         else:
#             temp += 1
#
#     sample_test = np.concatenate((sample_test_ictal, sample_test_interictal), axis=0)
#     label_test = np.concatenate((label_test_ictal, label_test_interictal), axis=0)
#
#     # 设置训练集和评估集
#     sample_train_ictal_temp = []
#     label_train_ictal_temp = []
#     sample_train_interictal_temp = []
#     label_train_interictal_temp = []
#
#     for index, sample in enumerate(sample_train_ictal):
#         if index == 0:
#             sample_train_ictal_temp = sample
#         else:
#             sample_train_ictal_temp = np.concatenate((sample_train_ictal_temp, sample), axis=0)
#     for index, label in enumerate(label_train_ictal):
#         if index == 0:
#             label_train_ictal_temp = label
#         else:
#             label_train_ictal_temp = np.concatenate((label_train_ictal_temp, label), axis=0)
#     for index, sample in enumerate(sample_train_interictal):
#         if index == 0:
#             sample_train_interictal_temp = sample
#         else:
#             sample_train_interictal_temp = np.concatenate((sample_train_interictal_temp, sample), axis=0)
#     for index, label in enumerate(label_train_interictal):
#         if index == 0:
#             label_train_interictal_temp = label
#         else:
#             label_train_interictal_temp = np.concatenate((label_train_interictal_temp, label), axis=0)
#
#     len_ictal = len(label_train_ictal_temp)
#     len_interictal = len(label_train_interictal_temp)
#
#     # 下采样使得ictal与interictal数据平衡
#     down_sampling_rate = int(np.floor(len_interictal / len_ictal))
#     if down_sampling_rate > 1:
#         sample_train_interictal_temp = sample_train_interictal_temp[::down_sampling_rate]
#         label_train_interictal_temp = label_train_interictal_temp[::down_sampling_rate]
#     elif down_sampling_rate == 1:
#         sample_train_interictal_temp = sample_train_interictal_temp[:len_ictal]
#         label_train_interictal_temp = label_train_interictal_temp[:len_ictal]
#
#     # 设置评估集
#     sample_train_ictal = sample_train_ictal_temp[: int(len_ictal * (1 - val_ratio))]
#     label_train_ictal = label_train_ictal_temp[: int(len_ictal * (1 - val_ratio))]
#     sample_train_interictal = sample_train_interictal_temp[: int(len_ictal * (1 - val_ratio))]
#     label_train_interictal = label_train_interictal_temp[: int(len_ictal * (1 - val_ratio))]
#
#     sample_train = np.concatenate((sample_train_ictal, sample_train_interictal), axis=0)
#     label_train = np.concatenate((label_train_ictal, label_train_interictal), axis=0)
#     label_train[label_train == 2] = 1
#
#     sample_val_ictal = sample_train_ictal_temp[int(len_ictal * (1 - val_ratio)):]
#     label_val_ictal = label_train_ictal_temp[int(len_ictal * (1 - val_ratio)):]
#     sample_val_interictal = sample_train_interictal_temp[int(len_ictal * (1 - val_ratio)):]
#     label_val_interictal = label_train_interictal_temp[int(len_ictal * (1 - val_ratio)):]
#
#     sample_val = np.concatenate((sample_val_ictal, sample_val_interictal), axis=0)
#     label_val = np.concatenate((label_val_ictal, label_val_interictal), axis=0)
#     label_val[label_val == 2] = 1
#
#     return sample_train, label_train, sample_val, label_val, sample_test, label_test


def train_test_split(preictal_x, preictal_y, interictal_x, interictal_y, order):

    num_preictal = len(preictal_x)

    # for some patients the interictal parts' number is different from the preictal parts' number

    if isinstance(interictal_x, list):
        # preictal_x = np.concatenate(preictal_x, axis=0)
        # preictal_y = np.concatenate(preictal_y, axis=0)
        interictal_x = np.concatenate(interictal_x, axis=0)
        interictal_y = np.concatenate(interictal_y, axis=0)

        # # shffule the interictal period
        # np.random.shuffle(preictal_x)
        # np.random.shuffle(interictal_x)

    # pre_folder_len = int(len(preictal_x) / num_preictal)
    inter_folder_len = int(len(interictal_x) / num_preictal)

    # pre_samples = []
    # pre_labels = []
    # for i in range(num_preictal):
    #     pre_samples.append(preictal_x[i * pre_folder_len: (i + 1) * pre_folder_len])
    #     pre_labels.append(preictal_y[i * pre_folder_len: (i + 1) * pre_folder_len])
    # preictal_x = pre_samples
    # preictal_y = pre_labels
    #
    inter_samples = []
    inter_labels = []
    for i in range(num_preictal):
        inter_samples.append((interictal_x[i * inter_folder_len: (i + 1) * inter_folder_len]))
        inter_labels.append((interictal_y[i * inter_folder_len: (i + 1) * inter_folder_len]))
    interictal_x = inter_samples
    interictal_y = inter_labels

    # 测试集
    test_preictal_x = preictal_x[order]
    test_preictal_y = preictal_y[order]
    test_interictal_x = interictal_x[order]
    test_interictal_y = interictal_y[order]

    # 删除测试集中label=2的label和sample
    # temp = 0
    # for i in range(len(test_preictal_y)):
    #     if test_preictal_y[temp] == 2:
    #         test_preictal_y.pop(temp)
    #         test_preictal_x.pop(temp)
    #     else:
    #         temp += 1
    test_preictal_x = np.array(test_preictal_x)
    test_preictal_y = np.array(test_preictal_y)

    test_preictal_x = test_preictal_x[test_preictal_y != 2]
    test_preictal_y = test_preictal_y[test_preictal_y != 2]

    test_preictal_x = test_preictal_x.tolist()
    test_preictal_y = test_preictal_y.tolist()

    test_sample = np.concatenate((test_preictal_x, test_interictal_x), axis=0)
    test_label = np.concatenate((test_preictal_y, test_interictal_y), axis=0)

    # 训练集
    train_preictal_x = []
    train_preictal_y = []
    train_interictal_x = []
    train_interictal_y = []

    # 测试集在 第一位，中间，最后 这三种情况（推出训练集的三种情况）
    for i in range(order):
        if i == 0:
            train_preictal_x = preictal_x[i]
            train_preictal_y = preictal_y[i]
            train_interictal_x = interictal_x[i]
            train_interictal_y = interictal_y[i]
        else:
            train_preictal_x = np.concatenate((train_preictal_x, preictal_x[i]), axis=0)
            train_preictal_y = np.concatenate((train_preictal_y, preictal_y[i]), axis=0)
            train_interictal_x = np.concatenate((train_interictal_x, interictal_x[i]), axis=0)
            train_interictal_y = np.concatenate((train_interictal_y, interictal_y[i]), axis=0)

    for i in range(order+1, len(preictal_x)):
        if len(train_preictal_x):
            train_preictal_x = np.concatenate((train_preictal_x, preictal_x[i]), axis=0)
            train_preictal_y = np.concatenate((train_preictal_y, preictal_y[i]), axis=0)
            train_interictal_x = np.concatenate((train_interictal_x, interictal_x[i]), axis=0)
            train_interictal_y = np.concatenate((train_interictal_y, interictal_y[i]), axis=0)
        else:
            if i == order+1:
                train_preictal_x = preictal_x[i]
                train_preictal_y = preictal_y[i]
                train_interictal_x = interictal_x[i]
                train_interictal_y = interictal_y[i]
            else:
                train_preictal_x = np.concatenate((train_preictal_x, preictal_x[i]), axis=0)
                train_preictal_y = np.concatenate((train_preictal_y, preictal_y[i]), axis=0)
                train_interictal_x = np.concatenate((train_interictal_x, interictal_x[i]), axis=0)
                train_interictal_y = np.concatenate((train_interictal_y, interictal_y[i]), axis=0)

    # 下采样使得训练集中preictal与interictal数据平衡
    len_preictal = len(train_preictal_y)
    len_interictal = len(train_interictal_y)
    down_sampling_rate = int(np.floor(len_interictal / len_preictal))

    # 2022.3.23
    np.random.shuffle(train_interictal_x)

    if down_sampling_rate > 1:
        train_interictal_x = train_interictal_x[::down_sampling_rate]
        train_interictal_y = train_interictal_y[::down_sampling_rate]
    elif down_sampling_rate == 1:
        train_interictal_x = train_interictal_x[:len_preictal]
        train_interictal_y = train_interictal_y[:len_preictal]

    # train_sample = np.concatenate((train_preictal_x, train_interictal_x), axis=0)
    # train_label = np.concatenate((train_preictal_y, train_interictal_y), axis=0)
    # train_label[train_label == 2] = 1

    # 标签为2的改为1 （标签为2的是重采样的数据）
    train_preictal_y[train_preictal_y == 2] = 1

    return train_preictal_x, train_preictal_y, train_interictal_x, train_interictal_y, test_sample, test_label


def train_val_split(pre_samples, pre_labels, inter_samples, inter_labels, train_bs, val_bs,):

    # # the code which using the original data
    # # 转置后两个维度, 并添加一个维度（为了输入数据）
    # pre_samples = pre_samples.transpose(0, 2, 1)
    # inter_samples = inter_samples.transpose(0, 2, 1)
    # pre_samples = Add_channel(pre_samples)
    # inter_samples = Add_channel(inter_samples)

    # use the source data (for our proposed method and the d_mlp)
    pre_samples = pre_samples[:, :, np.newaxis, :]
    inter_samples = inter_samples[:, :, np.newaxis, :]
    # # use the source data (for CNN method)
    # pre_samples = pre_samples[:, np.newaxis, :, :]
    # inter_samples = inter_samples[:, np.newaxis, :, :]

    np.random.shuffle(pre_samples)
    np.random.shuffle(inter_samples)

    pre_samples = torch.from_numpy(pre_samples)
    # # only use for chbmit22
    # inter_samples = np.array(inter_samples)
    inter_samples = torch.from_numpy(inter_samples)

    pre_labels = torch.from_numpy(pre_labels)
    # # only use for chbmit22
    # inter_labels = np.array(inter_labels)
    inter_labels = torch.from_numpy(inter_labels)

    # 测试集 比例：0.25
    val_pre = int(len(pre_samples) * 0.15)
    val_inter = int(len(inter_samples) * 0.15)

    val_samples = torch.cat((pre_samples[-val_pre:], inter_samples[-val_inter:]), dim=0)
    val_labels = torch.cat((pre_labels[-val_pre:], inter_labels[-val_inter:]), dim=0)

    val_dataset = data.TensorDataset(val_samples, val_labels)
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=val_bs,
        shuffle=True,
        num_workers=0,
    )

    # 训练集 比例：0.75
    train_samples = torch.cat((pre_samples[:-val_pre], inter_samples[:-val_inter]), dim=0)
    train_lables = torch.cat((pre_labels[:-val_pre], inter_labels[:-val_inter]), dim=0)

    train_dataset = data.TensorDataset(train_samples, train_lables)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=0,
    )

    return train_loader, val_loader


def train_val_split_lstm(pre_samples, pre_labels, inter_samples, inter_labels, train_bs, val_bs):

    np.random.shuffle(pre_samples)
    np.random.shuffle(inter_samples)

    pre_samples = torch.from_numpy(pre_samples)
    # # only use for chbmit22
    # inter_samples = np.array(inter_samples)
    inter_samples = torch.from_numpy(inter_samples)

    pre_labels = torch.from_numpy(pre_labels)
    # # only use for chbmit22
    # inter_labels = np.array(inter_labels)
    inter_labels = torch.from_numpy(inter_labels)

    # 测试集 比例：0.25
    val_pre = int(len(pre_samples) * 0.15)
    val_inter = int(len(inter_samples) * 0.15)

    val_samples = torch.cat((pre_samples[-val_pre:], inter_samples[-val_inter:]), dim=0)
    val_labels = torch.cat((pre_labels[-val_pre:], inter_labels[-val_inter:]), dim=0)

    val_dataset = data.TensorDataset(val_samples, val_labels)
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=val_bs,
        shuffle=True,
        num_workers=0,
    )

    # 训练集 比例：0.75
    train_samples = torch.cat((pre_samples[:-val_pre], inter_samples[:-val_inter]), dim=0)
    train_lables = torch.cat((pre_labels[:-val_pre], inter_labels[:-val_inter]), dim=0)

    train_dataset = data.TensorDataset(train_samples, train_lables)
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=0,
    )

    return train_loader, val_loader


def todataloader(sample, label, batch_size=4, shuffle=True):

    sample = torch.from_numpy(sample)
    label = torch.from_numpy(label)

    dataset = data.TensorDataset(sample, label)

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )

    return data_loader


def Add_channel(sample):

    sample_addc = sample[:, np.newaxis, :, :]

    return sample_addc



"""
the code which using the original data
"""
# def train_test_split(preictal_x, preictal_y, interictal_x, interictal_y, order):
#
#     num_preictal = len(preictal_x)
#     num_interictal = len(interictal_x)
#
#     if num_interictal > num_preictal:
#         if isinstance(interictal_x, list):
#             interictal_x = np.concatenate(interictal_x, axis=0)
#             interictal_y = np.concatenate(interictal_y, axis=0)
#     interictal_folder_len = int(len(interictal_x) / num_preictal)
#
#     # for some patients data the interictal parts' number will bigger than the preictal parts' number
#     if num_interictal > num_preictal:
#         temp1_samples = []
#         temp1_labels = []
#         for i in range(num_preictal):
#             temp2_samples = []
#             temp2_labels = []
#             for j in range(interictal_folder_len):
#                 temp2_samples.append(interictal_x[(i * interictal_folder_len) + j])
#                 temp2_labels.append(interictal_y[(i * interictal_folder_len) + j])
#             temp1_samples.append(temp2_samples)
#             temp1_labels.append(temp2_labels)
#         interictal_x = temp1_samples
#         interictal_y = temp1_labels
#
#
#     # 测试集
#     test_preictal_x = preictal_x[order]
#     test_preictal_y = preictal_y[order]
#     test_interictal_x = interictal_x[order]
#     test_interictal_y = interictal_y[order]
#
#     # 删除测试集中label=2的label和sample
#     # temp = 0
#     # for i in range(len(test_preictal_y)):
#     #     if test_preictal_y[temp] == 2:
#     #         test_preictal_y.pop(temp)
#     #         test_preictal_x.pop(temp)
#     #     else:
#     #         temp += 1
#     test_preictal_x = np.array(test_preictal_x)
#     test_preictal_y = np.array(test_preictal_y)
#
#     test_preictal_x = test_preictal_x[test_preictal_y != 2]
#     test_preictal_y = test_preictal_y[test_preictal_y != 2]
#
#     test_preictal_x = test_preictal_x.tolist()
#     test_preictal_y = test_preictal_y.tolist()
#
#     test_sample = np.concatenate((test_preictal_x, test_interictal_x), axis=0)
#     test_label = np.concatenate((test_preictal_y, test_interictal_y), axis=0)
#
#     # 训练集
#     train_preictal_x = []
#     train_preictal_y = []
#     train_interictal_x = []
#     train_interictal_y = []
#
#     # 测试集在 第一位，中间，最后 这三种情况（推出训练集的三种情况）
#     for i in range(order):
#         if i == 0:
#             train_preictal_x = preictal_x[i]
#             train_preictal_y = preictal_y[i]
#             train_interictal_x = interictal_x[i]
#             train_interictal_y = interictal_y[i]
#         else:
#             train_preictal_x = np.concatenate((train_preictal_x, preictal_x[i]), axis=0)
#             train_preictal_y = np.concatenate((train_preictal_y, preictal_y[i]), axis=0)
#             train_interictal_x = np.concatenate((train_interictal_x, interictal_x[i]), axis=0)
#             train_interictal_y = np.concatenate((train_interictal_y, interictal_y[i]), axis=0)
#
#     for i in range(order+1, len(preictal_x)):
#         if len(train_preictal_x):
#             train_preictal_x = np.concatenate((train_preictal_x, preictal_x[i]), axis=0)
#             train_preictal_y = np.concatenate((train_preictal_y, preictal_y[i]), axis=0)
#             train_interictal_x = np.concatenate((train_interictal_x, interictal_x[i]), axis=0)
#             train_interictal_y = np.concatenate((train_interictal_y, interictal_y[i]), axis=0)
#         else:
#             if i == order+1:
#                 train_preictal_x = preictal_x[i]
#                 train_preictal_y = preictal_y[i]
#                 train_interictal_x = interictal_x[i]
#                 train_interictal_y = interictal_y[i]
#             else:
#                 train_preictal_x = np.concatenate((train_preictal_x, preictal_x[i]), axis=0)
#                 train_preictal_y = np.concatenate((train_preictal_y, preictal_y[i]), axis=0)
#                 train_interictal_x = np.concatenate((train_interictal_x, interictal_x[i]), axis=0)
#                 train_interictal_y = np.concatenate((train_interictal_y, interictal_y[i]), axis=0)
#
#     # 下采样使得训练集中preictal与interictal数据平衡
#     len_preictal = len(train_preictal_y)
#     len_interictal = len(train_interictal_y)
#     down_sampling_rate = int(np.floor(len_interictal / len_preictal))
#
#     if down_sampling_rate > 1:
#         train_interictal_x = train_interictal_x[::down_sampling_rate]
#         train_interictal_y = train_interictal_y[::down_sampling_rate]
#     elif down_sampling_rate == 1:
#         train_interictal_x = train_interictal_x[:len_preictal]
#         train_interictal_y = train_interictal_y[:len_preictal]
#
#     # train_sample = np.concatenate((train_preictal_x, train_interictal_x), axis=0)
#     # train_label = np.concatenate((train_preictal_y, train_interictal_y), axis=0)
#     # train_label[train_label == 2] = 1
#
#     # 标签为2的改为1 （标签为2的是重采样的数据）
#     train_preictal_y[train_preictal_y == 2] = 1
#
#     return train_preictal_x, train_preictal_y, train_interictal_x, train_interictal_y, test_sample, test_label
#
#
# def train_val_split(pre_samples, pre_labels, inter_samples, inter_labels, train_bs, val_bs,):
#
#     # 转置后两个维度, 并添加一个维度（为了输入数据）
#     pre_samples = pre_samples.transpose(0, 2, 1)
#     inter_samples = inter_samples.transpose(0, 2, 1)
#
#     pre_samples = Add_channel(pre_samples)
#     inter_samples = Add_channel(inter_samples)
#
#     pre_samples = torch.from_numpy(pre_samples)
#     inter_samples = torch.from_numpy(inter_samples)
#
#     pre_labels = torch.from_numpy(pre_labels)
#     inter_labels = torch.from_numpy(inter_labels)
#
#     # 测试集 比例：0.25
#     val_pre = int(len(pre_samples) * 0.25)
#     val_inter = int(len(inter_samples) * 0.25)
#
#     val_samples = torch.cat((pre_samples[-val_pre:], inter_samples[-val_inter:]), dim=0)
#     val_labels = torch.cat((pre_labels[-val_pre:], inter_labels[-val_inter:]), dim=0)
#
#     val_dataset = data.TensorDataset(val_samples, val_labels)
#     val_loader = data.DataLoader(
#         dataset=val_dataset,
#         batch_size=val_bs,
#         shuffle=False,
#         num_workers=2,
#     )
#
#     # 训练集 比例：0.75
#     train_samples = torch.cat((pre_samples[:-val_pre], inter_samples[:-val_inter]), dim=0)
#     train_lables = torch.cat((pre_labels[:-val_pre], inter_labels[:-val_inter]), dim=0)
#
#     train_dataset = data.TensorDataset(train_samples, train_lables)
#     train_loader = data.DataLoader(
#         dataset=train_dataset,
#         batch_size=train_bs,
#         shuffle=True,
#         num_workers=2,
#     )
#
#     return train_loader, val_loader
#
#
# def todataloader(sample, label, batch_size=4):
#
#     # https://www.cnblogs.com/demo-deng/p/10623334.html https://blog.csdn.net/cxkshizhenshuai/article/details/119182275
#
#     sample = torch.from_numpy(sample)
#     label = torch.from_numpy(label)
#
#     dataset = data.TensorDataset(sample, label)
#
#     if batch_size == 4:
#         data_loader = data.DataLoader(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=2,
#         )
#
#         return data_loader
#
#     elif batch_size == 1:
#         data_loader = data.DataLoader(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=2,
#         )
#
#         return data_loader
#
#     else:
#         print('batch_size = 4 for train, batch_size = 1 for test or validation')
#         exit(1)
#
#
#
#
# def Add_channel(sample):
#
#     sample_addc = sample[:, np.newaxis, :, :]
#
#     return sample_addc