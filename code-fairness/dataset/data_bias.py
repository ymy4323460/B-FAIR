import numpy as np
import pandas as pd
from scipy.stats import binom
import torch
import os


def user_item_feature_withconfounder(context_dim=32, sample_size=128, item_size=100):
    confounder = np.random.uniform(-1, 1, (sample_size, context_dim))
    x = np.random.uniform(-1, 1, (sample_size, context_dim))
    a = np.random.uniform(-1, 1, (item_size, context_dim))
    # x = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, context_dim))
    # a = np.random.normal(loc=0.0, scale=1.0, size=(item_size, context_dim))
    return x, a, confounder


def linear_impression_list(context_dim=32, impression_len=8, sample_size=128, item_size=16, step=10, mode='dev'):
    x, a = user_item_feature(context_dim=context_dim, sample_size=sample_size, item_size=item_size)

    user_feature_dict = dict(zip([j for j in range(sample_size)], list(x)))
    item_feature_dict = dict(zip([j for j in range(item_size)], list(a)))
    np.save('./lineardata/user', x)
    np.save('./lineardata/item', a)
    np.save('./lineardata/user_features', user_feature_dict)
    np.save('./lineardata/item_features', item_feature_dict)
    all_ = None
    for i in range(step):
        pair_matrix = torch.from_numpy(
            np.dot(x, a.T) + np.random.normal(loc=0.0, scale=0.01, size=(sample_size, item_size)))
        value, index = torch.topk(pair_matrix, impression_len, dim=1, largest=True, sorted=True, out=None)

        user_feedback = np.zeros([sample_size, impression_len])
        user_feedback[:, 0] = 1
        impression_list = index.numpy()
        assert impression_list.shape == user_feedback.shape
        user_list = np.repeat(np.arange(sample_size).reshape([sample_size, 1]), impression_len, axis=1).reshape(-1, 1)
        impression_list = impression_list.reshape(-1, 1)
        impression_indicate = np.ones([sample_size, impression_len]).reshape(-1, 1)
        user_feedback = user_feedback.reshape(-1, 1)
        assert user_list.shape == impression_list.shape
        batch = np.concatenate((user_list, impression_list, user_feedback, impression_indicate), axis=1)
        if all_ is None:
            all_ = batch
        else:
            all_ = np.concatenate((all_, batch), axis=0)
    pd.DataFrame(all_).to_csv('./lineardata/{}/data_nonuniform.csv'.format(mode), header=False, index=False)
    return all_, user_feature_dict, item_feature_dict


def get_impression_feature(x, parameters=None):
    item_feature_dict = np.load(
        './nonliearlogitdata_{}_{}_{}_{}/item_features.npy'.format(parameters[1], parameters[2], parameters[3],
                                                                   parameters[4]), allow_pickle=True).item()
    # print(item_feature_dict)
    # item_feature_dict = np.load('./lineardata/item_feature_dict.npy', allow_pickle=True).item()
    return item_feature_dict[x]


def cal_similarity(x, y):
    # print(np.dot(x.reshape([1, x.shape[0]]), y.T).shape)
    return np.dot(x.reshape([1, x.shape[0]]), y.T)


def sigmoid_fun(x):
    return 1 / (1 + np.exp(-x))


def cal_similarity_uniform(x, y):
    # print(np.dot(x.reshape([1, x.shape[0]]), y.T).shape)
    return np.where(sigmoid_fun(np.dot(x.reshape([1, x.shape[0]]), y.T)) > 0.5, 1, 0)


def logit_policy(x, y, confounder, gamma=0.5):
    # print(x)
    #
    # print(y)
    x = x.reshape(1, x.shape[0])
    all_x = (1 - gamma) * np.concatenate((np.tile(x, (y.shape[0], 1)), y), axis=1) + gamma * np.tile(confounder,
                                                                                                     (y.shape[0], 2))
    indicate = np.where(all_x > 0, 1, 0)
    # print(indicate)
    # result = np.sum(np.multiply(all_x, indicate)-0.5*indicate, axis=1)
    result = np.sum(all_x, axis=1)
    # print(result)
    # print(sigmoid_fun(result))
    # exit()
    return sigmoid_fun(result)


def nonlinear_reward_function(x, y, confounder, gamma=0.5):
    x = x.reshape(1, x.shape[0])
    all_x = np.concatenate((np.tile(x, (y.shape[0], 1)), y), axis=1)  # + gamma*np.tile(confounder, (y.shape[0],2))
    indicate = np.where(all_x > 0, 1, 0)
    indicate_neg = np.where(all_x < 0, 1, 0)
    pos = np.multiply(all_x, indicate) - 0.5 * indicate
    neg = np.multiply(all_x, indicate_neg) + 0.5 * indicate
    # print(np.mean(np.matmul(pos.reshape(all_x.shape[0], all_x.shape[1], 1), neg.reshape(all_x.shape[0], 1, all_x.shape[1])), axis=1).shape)
    # print(np.mean(pos, axis=1).shape)
    result = np.sum(pos, axis=1) + (1 / 32.) * np.sum(
        np.matmul(pos.reshape(all_x.shape[0], all_x.shape[1], 1), neg.reshape(all_x.shape[0], 1, all_x.shape[1])),
        axis=(1, 2))
    # print(result)
    # exit()
#     result = gamma*confounder + (1-gamma)*result
    return np.where(sigmoid_fun(result) > 0.5, 1, 0)


def nonlinear_reward_function_logit(x, y, confounder, gamma=0.5):
    x = x.reshape(1, x.shape[0])
    # x = (1-gamma)*x.reshape(1, x.shape[0])+gamma*confounder.reshape(1, x.shape[0])
    # x = x.reshape(1, x.shape[0])
    all_x = np.concatenate((np.tile(x, (y.shape[0], 1)), y), axis=1)  # + gamma*np.tile(confounder, (y.shape[0],2))
    indicate = np.where(all_x > 0, 1, 0)
    indicate_neg = np.where(all_x < 0, 1, 0)
    pos = np.multiply(all_x, indicate) - 0.5 * indicate
    neg = np.multiply(all_x, indicate_neg) + 0.5 * indicate
    # print(np.mean(np.matmul(pos.reshape(all_x.shape[0], all_x.shape[1], 1), neg.reshape(all_x.shape[0], 1, all_x.shape[1])), axis=1).shape)
    # print(np.mean(pos, axis=1).shape)
    result = np.sum(pos, axis=1) + (1 / 32.) * np.sum(
        np.matmul(pos.reshape(all_x.shape[0], all_x.shape[1], 1), neg.reshape(all_x.shape[0], 1, all_x.shape[1])),
        axis=(1, 2))
    # print(result)
    # exit()
    return sigmoid_fun(result)


def linear_reward_function(x, y):
    # print(np.dot(x.reshape([1, x.shape[0]]), y.T).shape)
    return np.where(sigmoid_fun(np.dot(x.reshape([1, x.shape[0]]), y.T)) > 0.5, 1, 0)


def sample_from_multinomial(probability, bs=0.2, impression_len=5, item_size=32):
    # print(probability)
    probability = (1 - bs) * (
                probability + np.abs(np.random.normal(loc=0.0, scale=0.02, size=(probability.shape)))) + bs * (
                              1. / item_size * np.ones(item_size))
    return np.random.choice(np.arange(item_size), size=impression_len, replace=False, p=probability / probability.sum())
    # return np.random.multinomial(impression_len, probability/probability.sum(), size=1)


def random_policy(impression_len=5, item_size=32):
    # print(probability)
    probability = 1. / impression_len * np.ones(item_size)
    return np.random.choice(np.arange(item_size), size=impression_len, replace=False, p=probability / probability.sum())
    # return np.random.multinomial(impression_len, probability/probability.sum(), size=1)


def get_feedbacks(x, y, threshold=None):
    if threshold is not None:
        # print(threshold)
        for i in range(threshold.shape[0]):
            if threshold[i] == 1:
                x[y[i]] = 1
    else:
        x[y] = 1
    return x


def logit_impression_list_new(context_dim=32, impression_len=5, sample_size=128, item_size=32, step=10, mode='dev',
                              sample_num=50, policy='logit', gamma=0.5, bs=0.01):
    sample_size = sample_num * 200
    x, a, confounder = user_item_feature_withconfounder(context_dim=context_dim, sample_size=sample_num * 200,
                                                        item_size=item_size)
    if mode == 'train':
        x = np.load('./{}data_{}_{}_{}_{}/user.npy'.format(policy, sample_num, context_dim, gamma, bs),
                    allow_pickle=True)
        # print(x)
        a = np.load('./{}data_{}_{}_{}_{}/item.npy'.format(policy, sample_num, context_dim, gamma, bs),
                    allow_pickle=True)
        confounder = np.load('./{}data_{}_{}_{}_{}/confounder.npy'.format(policy, sample_num, context_dim, gamma, bs),
                             allow_pickle=True)
    user_feature_dict = dict(zip([j for j in range(sample_size)], list(x)))
    item_feature_dict = dict(zip([j for j in range(item_size)], list(a)))
    # print(x)
    if not os.path.exists('./nonliearlogitdata_{}_{}_{}_{}/'.format(sample_num, context_dim, gamma, bs)):  #
        os.makedirs('./nonliearlogitdata_{}_{}_{}_{}/dev/'.format(sample_num, context_dim, gamma, bs))
        os.makedirs('./nonliearlogitdata_{}_{}_{}_{}/train/'.format(sample_num, context_dim, gamma, bs))
    np.save('./nonliearlogitdata_{}_{}_{}_{}/user'.format(sample_num, context_dim, gamma, bs), x)
    np.save('./nonliearlogitdata_{}_{}_{}_{}/item'.format(sample_num, context_dim, gamma, bs), a)
    np.save('./nonliearlogitdata_{}_{}_{}_{}/confounder'.format(sample_num, context_dim, gamma, bs), confounder)
    np.save('./nonliearlogitdata_{}_{}_{}_{}/user_features'.format(sample_num, context_dim, gamma, bs),
            user_feature_dict)
    np.save('./nonliearlogitdata_{}_{}_{}_{}/item_features'.format(sample_num, context_dim, gamma, bs),
            item_feature_dict)
    all_ = None
    for i in range(1):
        # impression_p = 1/(1+np.exp(-np.dot(x, a.T) ))#np.random.normal(loc=0.0, scale=0.01, size=(sample_size, item_size))
        # impression_p = 1./16*np.ones((sample_size, item_size))#np.array(list(map(logit_policy, x, [a for j in range(sample_size)])))
        impression_p = np.array(list(
            map(logit_policy, x, [a for j in range(sample_size)], confounder, [gamma for j in range(sample_size)])))

        # print(impression_p.shape)
        #
        # print(impression_p[0].sum())
        impression_list = np.array(
            list(map(sample_from_multinomial, impression_p, [bs for j in range(impression_p.shape[0])])))
        # print(impression_list)
        impression_information = np.array(list(map(get_impression_feature, impression_list.reshape(-1),
                                                   [['logit', sample_num, context_dim, gamma, bs] for j in
                                                    range(sample_size * impression_len)]))).reshape(
            [sample_size, impression_len, context_dim])
        # print(impression_list.reshape(-1))
        #
        pair_matrix = np.array(list(map(nonlinear_reward_function_logit, x, impression_information, confounder,
                                        [gamma for j in range(sample_size)]))).reshape(sample_size, impression_len)
        # print(pair_matrix, pair_matrix.shape)
        value, index = torch.topk(torch.from_numpy(pair_matrix), 2, dim=1, largest=True, sorted=True, out=None)[
                           0].numpy(), \
                       torch.topk(torch.from_numpy(pair_matrix), 2, dim=1, largest=True, sorted=True, out=None)[
                           1].numpy()
        # print(value)
        # value = np.where(value > 0.5, 1, 0)#
        # user_feedback = np.zeros([sample_size, impression_len]) #
        # assert index.shape[0] == user_feedback.shape[0]
        user_feedback = np.where(pair_matrix > 0.5, 1, 0)
        # user_feedback = np.array(list(map(get_feedbacks, user_feedback, index, value)))
        # print(user_feedback)

        assert impression_list.shape == user_feedback.shape
        user_list = np.repeat(np.arange(sample_size).reshape([sample_size, 1]), impression_len, axis=1).reshape(-1, 1)
        # print(user_list)
        impression_list = impression_list.reshape(-1, 1)
        impression_indicate = np.ones([sample_size, impression_len]).reshape(-1, 1)
        user_feedback = user_feedback.reshape(-1, 1)
        assert user_list.shape == impression_list.shape
        # print(user_list.shape, impression_list.shape, impression_indicate.shape, user_feedback.shape)
        batch = np.concatenate((user_list, impression_list, user_feedback, impression_indicate), axis=1)
        if all_ is None:
            all_ = batch
        else:
            all_ = np.concatenate((all_, batch), axis=0)
        # print(batch)
    if not os.path.exists('./nonliearlogitdata_{}_{}_{}_{}/'.format(sample_num, context_dim, gamma, bs)):  #
        os.makedirs('./nonliearlogitdata_{}_{}_{}_{}/dev/'.format(sample_num, context_dim, gamma, bs))
        os.makedirs('./nonliearlogitdata_{}_{}_{}_{}/train/'.format(sample_num, context_dim, gamma, bs))
    partition = int(0.75 * all_.shape[0])
    pd.DataFrame(all_[: partition]).to_csv(
        './nonliearlogitdata_{}_{}_{}_{}/dev/data_nonuniform.csv'.format(sample_num, context_dim, gamma, bs),
        header=False, index=False)
    pd.DataFrame(all_[partition:]).to_csv(
        './nonliearlogitdata_{}_{}_{}_{}/train/data_nonuniform.csv'.format(sample_num, context_dim, gamma, bs),
        header=False, index=False)
    return all_, user_feature_dict, item_feature_dict


def random_impression_list(context_dim=32, impression_len=10, sample_size=128, item_size=32, step=10, mode='dev',
                           policy='logit', sample_num=50, gamma=0.5, bs=0.01):
    sample_size = sample_num * 200
    x = np.load('./{}data_{}_{}_{}_{}/user.npy'.format(policy, sample_num, context_dim, gamma, bs), allow_pickle=True)
    # print(x)_{}
    a = np.load('./{}data_{}_{}_{}_{}/item.npy'.format(policy, sample_num, context_dim, gamma, bs), allow_pickle=True)
    confounder = np.load('./{}data_{}_{}_{}_{}/confounder.npy'.format(policy, sample_num, context_dim, gamma, bs),
                         allow_pickle=True)
    all_ = None
    for i in range(1):
        #
        # impression_list = np.random.randint(0, high = item_size, size = (sample_size, impression_len), dtype = 'l')
        impression_list = np.array(list(map(random_policy, [impression_len for j in range(sample_size)])))
        #
        # print(impression_list)
        impression_information = np.array(list(map(get_impression_feature, impression_list.reshape(-1),
                                                   [[policy, sample_num, context_dim, gamma, bs] for j in
                                                    range(sample_size * impression_len)]))).reshape(
            [sample_size, impression_len, context_dim])
        # print(x.shape)
        #
        pair_matrix = np.array(list(map(nonlinear_reward_function, x, impression_information, confounder,
                                        [gamma for j in range(sample_size)]))).reshape(impression_list.shape)
        # print(pair_matrix.shape, impression_list.shape)
        # index = torch.topk(torch.from_numpy(pair_matrix), 1, dim=1, largest=True, sorted=True, out=None)[1].numpy()
        user_feedback = pair_matrix
        # assert index.shape[0] == user_feedback.shape[0]

        # user_feedback = np.array(list(map(get_feedbacks, user_feedback, index)))

        assert impression_list.shape == user_feedback.shape
        user_list = np.repeat(np.arange(sample_size).reshape([sample_size, 1]), impression_len, axis=1).reshape(-1, 1)
        impression_list = impression_list.reshape(-1, 1)
        impression_indicate = np.ones([sample_size, impression_len]).reshape(-1, 1)
        user_feedback = user_feedback.reshape(-1, 1)
        assert user_list.shape == impression_list.shape
        batch = np.concatenate((user_list, impression_list, user_feedback, impression_indicate), axis=1)
        if all_ is None:
            all_ = batch
        else:
            all_ = np.concatenate((all_, batch), axis=0)

    partition = int(0.75 * all_.shape[0])
    pd.DataFrame(all_[: partition]).to_csv(
        './{}data_{}_{}_{}_{}/dev/data_uniform.csv'.format(policy, sample_num, context_dim, gamma, bs), header=False,
        index=False)
    pd.DataFrame(all_[partition:]).to_csv(
        './{}data_{}_{}_{}_{}/train/data_uniform.csv'.format(policy, sample_num, context_dim, gamma, bs), header=False,
        index=False)

    return all_  # , user_feature_dict, item_feature_dict


for sam in [10]:
    for cdim in [32]:
        for gm in [0.5]:
            for bs in [0.5]:
                logit_impression_list_new(mode='dev', step=sam, policy='nonliearlogit', sample_num=sam,
                                          context_dim=cdim, gamma=gm, bs=bs)
                # logit_impression_list_new(mode='train', step=10, sample_num=sam, context_dim=cdim)

                # linear_impression_list(mode='dev', step=30)
                # linear_impression_list(mode='train', step=5)

                random_impression_list(mode='dev', step=sam, policy='nonliearlogit', sample_num=sam, context_dim=cdim,
                                       gamma=gm, bs=bs)
        # random_impression_list(mode='train', step=10, policy='nonliearlogit', sample_num=sam, context_dim=cdim)

        # random_impression_list(mode='dev', step=30, policy='linear')
        # random_impression_list(mode='train', step=5, policy='linear')





