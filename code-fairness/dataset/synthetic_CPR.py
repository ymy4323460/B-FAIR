import numpy as np
import pandas as pd
import random
from scipy.stats import binom
# import torch
import os


def negative_sampling(item, reward, k):
    #     print('negative_sampling')
    #     print(data['reward'])
    impression_dic = dict(zip(list(item), list(reward)))
    item_list = [i for i in range(16)]

    impression_len = len(list(impression_dic.keys()))
    #     print(impression_len)
    for i in range(k - impression_len):
        print('negative_sampling')
        print(k, impression_len)
        a = random.choice(item_list)
        while a in list(impression_dic.keys()):
            a = random.choice(item_list)
        #         print(a)
        impression_dic[a] = 0
    new_impression = list(impression_dic.keys()) + list(impression_dic.values())
    #     print(new_impression)
    return new_impression


def save_bpr_data(data):  # ['user', 'item', 'impression', 'click', 'impression_id']
    print('save_bpr_data')
    prefer_df = data[data['click'] == 1]
    not_prefer_df = data[data['click'] == 0]
    prefer_df.columns = ['user', 'prefer_item', 'is_exposed', 'click', 'impression_id']
    not_prefer_df.columns = ['user', 'not_prefer_item', 'is_exposed', 'click', 'impression_id']

    merge_df = pd.merge(prefer_df, not_prefer_df, on=['user', 'impression_id'], how='right')

    merge_small = merge_df[['user', 'prefer_item', 'not_prefer_item']].dropna()
    #     print(merge_small)
    return merge_small


def save_s_model_data(user_list, impression_list, user_feedback, impression_len, k):
    print('save_s_model_data')
    # batch = np.concatenate((user_list, impression_list, impression_indicate, user_feedback, impression_id.reshape(-1,1)), axis=1)
    s_user_list = user_list.reshape(-1, impression_len)[:, 0]
    #     print(s_user_list)
    s_item_list = impression_list.reshape(-1, impression_len)
    s_reward_list = user_feedback.reshape(-1, impression_len)
    #     all_data = np.concatenate((s_user_list, s_item_list, s_reward_list), axis=1)
    #     all_data = pd.DataFrame(all_data)
    #     print(all_data)
    item_reward = np.array(
        list(map(negative_sampling, list(s_item_list), list(s_reward_list), [k for i in range(len(s_item_list))])))
    all_data = {
        "user": list(s_user_list),
        "item_reward": list(item_reward)
    }
    all_data = pd.DataFrame(all_data)
    #     print(item_reward.shape)
    final_df = all_data['item_reward'].apply(pd.Series, index=[i for i in range(1, 1 + 2 * k)])
    final_df['user'] = all_data['user']
    column = ['user'] + [i for i in range(1, 1 + 2 * k)]
    final_df = final_df[column]
    # print(final_df.values.shape)
    return final_df


def user_item_feature(context_dim=32, sample_size=128, item_size=100):
    x = np.random.uniform(-1, 1, (sample_size, context_dim))
    a = np.random.uniform(-1, 1, (item_size, context_dim))
    # x = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, context_dim))
    # a = np.random.normal(loc=0.0, scale=1.0, size=(item_size, context_dim))
    return x, a


def get_impression_feature(x, parameters=None):
    item_feature_dict = np.load(
        './synthetic/{}data_{}_{}/item_features.npy'.format(parameters[0], parameters[1], parameters[2]),
        allow_pickle=True).item()
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


# def logit_policy(x, y):
#     # print(x)
#     #
#     # print(y)

#     x = x.reshape(1, x.shape[0])
#     all_x = np.concatenate((np.tile(x, (y.shape[0],1)), y), axis=1)
#     indicate = np.where(all_x>0, 1, 0)
#     # print(indicate)
#     # result = np.sum(np.multiply(all_x, indicate)-0.5*indicate, axis=1)
#     result = np.sum(all_x, axis=1)
#     # print(result)
#     # print(sigmoid_fun(result))
#     # exit()
#     return sigmoid_fun(result)


# -----------------------------------------------------
# Quanyu Dai
def logit_policy(x, y):
    # print(x)
    # print('-----------------')
    # print(y)
    x = x.reshape(1, x.shape[0])
    x = np.tile(x, (y.shape[0], 1))
    # print('shape:', x.shape, y.shape)
    scores = np.multiply(x, y)
    result = np.sum(scores, axis=1)
    scores = sigmoid_fun(result)
    # print('-----------------')
    # print('result\n', result)
    # print('-----------------')
    # print('scores\n', scores)
    # print()

    return scores


# -----------------------------------------------------

def nonlinear_reward_function(x, y):
    x = x.reshape(1, x.shape[0])
    all_x = np.concatenate((np.tile(x, (y.shape[0], 1)), y), axis=1)
    indicate = np.where(all_x > 0, 1, 0)
    result = np.sum(np.multiply(all_x, indicate) - 0.5 * indicate, axis=1)
    return np.where(sigmoid_fun(result) > 0.5, 1, 0)


def nonlinear_reward_function_logit(x, y):
    x = x.reshape(1, x.shape[0])
    all_x = np.concatenate((np.tile(x, (y.shape[0], 1)), y), axis=1)
    indicate = np.where(all_x > 0, 1, 0)
    result = np.sum(np.multiply(all_x, indicate) - 0.5 * indicate, axis=1)
    return sigmoid_fun(result)


def linear_reward_function(x, y):
    # print(np.dot(x.reshape([1, x.shape[0]]), y.T).shape)
    return np.where(sigmoid_fun(np.dot(x.reshape([1, x.shape[0]]), y.T)) > 0.5, 1, 0)


def sample_from_multinomial(probability, impression_len=8, item_size=16):
    #     print(probability.shape)
#     print(probability.shape)
    probability = probability + np.abs(np.random.normal(loc=0.0, scale=0.2, size=(probability.shape)))
    return np.random.choice(np.arange(item_size), size=impression_len, replace=False, p=probability / probability.sum())
    # return np.random.multinomial(impression_len, probability/probability.sum(), size=1)


def random_policy(impression_len=8, item_size=16):
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


def logit_impression_list_new(context_dim=32, impression_len=5, sample_size=128, item_size=32, step=10, mode='train',
                              sample_num=50, k=5):

    # print(x)
    if not os.path.exists(
            './synthetic/logitdata_{}_{}/'.format(sample_num, context_dim)):  # 鍒ゆ柇鎵€鍦ㄧ洰褰曚笅鏄惁鏈夎鏂囦欢鍚嶇殑鏂囦欢澶?
        os.makedirs('./synthetic/logitdata_{}_{}/train/'.format(sample_num, context_dim))
        os.makedirs('./synthetic/logitdata_{}_{}/dev/'.format(sample_num, context_dim))
    if mode == 'train':
        x, a = user_item_feature(context_dim=context_dim, sample_size=sample_size, item_size=item_size)
        user_feature_dict = dict(zip([j for j in range(sample_size)], list(x)))
        item_feature_dict = dict(zip([j for j in range(item_size)], list(a)))
        np.save('./synthetic/logitdata_{}_{}/user'.format(sample_num, context_dim), x)
        np.save('./synthetic/logitdata_{}_{}/item'.format(sample_num, context_dim), a)
        np.save('./synthetic/logitdata_{}_{}/user_features'.format(sample_num, context_dim), user_feature_dict)
        np.save('./synthetic/logitdata_{}_{}/item_features'.format(sample_num, context_dim), item_feature_dict)
    else:
        x = np.load('./synthetic/logitdata_{}_{}/user.npy'.format(sample_num, context_dim), allow_pickle=True)
        a = np.load('./synthetic/logitdata_{}_{}/item.npy'.format(sample_num, context_dim), allow_pickle=True)
        user_feature_dict = np.load('./synthetic/logitdata_{}_{}/user_features.npy'.format(sample_num, context_dim), allow_pickle=True)
        item_feature_dict = np.load('./synthetic/logitdata_{}_{}/item_features.npy'.format(sample_num, context_dim), allow_pickle=True)

    all_ = None
    s_model_data = None
    for i in range(step):
        #         print(i)
        # impression_p = 1/(1+np.exp(-np.dot(x, a.T) ))#np.random.normal(loc=0.0, scale=0.01, size=(sample_size, item_size))
        # impression_p = 1./16*np.ones((sample_size, item_size))#np.array(list(map(logit_policy, x, [a for j in range(sample_size)])))
        impression_p = np.array(list(map(logit_policy, x, [a for j in range(sample_size)])))

#         print(impression_p.shape)
        # 鑾峰緱impression list 鐨刦eature
        # print(impression_p[0].sum())
        impression_list = np.array(list(
            map(sample_from_multinomial, impression_p, [impression_len for j in range(sample_size)],
                [item_size for j in range(sample_size)])))
        # print(impression_list)
        impression_information = np.array(list(map(get_impression_feature, impression_list.reshape(-1),
                                                   [['logit', sample_num, context_dim] for j in
                                                    range(sample_size * impression_len)]))).reshape(
            [sample_size, impression_len, context_dim])
        # print(impression_list.reshape(-1))
        # 鐢?true function 鑾峰緱 result
        pair_matrix = np.array(list(map(nonlinear_reward_function_logit, x, impression_information))).reshape(
            sample_size, impression_len)
        # print(pair_matrix, pair_matrix.shape)
        #         value, index = torch.topk(torch.from_numpy(pair_matrix), 2, dim=1, largest=True, sorted=True, out=None)[0].numpy(), \
        #                         torch.topk(torch.from_numpy(pair_matrix), 2, dim=1, largest=True, sorted=True, out=None)[1].numpy()
        # print(value)
        # value = np.where(value > 0.5, 1, 0)# 閲嶈
        # user_feedback = np.zeros([sample_size, impression_len]) #閲嶈
        # assert index.shape[0] == user_feedback.shape[0]
        user_feedback = np.where(pair_matrix > 0.5, 1, 0)

        # user_feedback = np.array(list(map(get_feedbacks, user_feedback, index, value))) #閲嶈
        # print(user_feedback)

        #         assert impression_list.shape == user_feedback.shape
        user_list = np.repeat(np.arange(sample_size).reshape([sample_size, 1]), impression_len, axis=1).reshape(-1, 1)
        #         print(user_list)
        impression_list = impression_list.reshape(-1, 1)
        impression_indicate = np.ones([sample_size, impression_len]).reshape(-1, 1)
        user_feedback = user_feedback.reshape(-1, 1)
        impression_id = i * sample_size + user_list

        #         assert user_list.shape == impression_list.shape
        #         print(user_list.shape, impression_list.shape, impression_indicate.shape)
        batch = np.concatenate(
            (user_list, impression_list, impression_indicate, user_feedback, impression_id.reshape(-1, 1)), axis=1)
        #         print(batch)
        if all_ is None:
            all_ = batch
        else:
            all_ = np.concatenate((all_, batch), axis=0)
    smodel_df = save_s_model_data(all_[:, 0], all_[:, 1], all_[:, 3], impression_len, k)

    all_ = pd.DataFrame(all_)

    all_.columns = ['user', 'item', 'impression', 'click', 'impression_id']
    bpr_df = save_bpr_data(all_)
    print(mode)

    pd.DataFrame(all_[['user', 'item', 'impression', 'click']]).to_csv(
        './synthetic/logitdata_{}_{}/{}/data.csv'.format(sample_num, context_dim, mode), header=False, index=False)
    pd.DataFrame(bpr_df).to_csv('./synthetic/logitdata_{}_{}/{}/bpr_data.csv'.format(sample_num, context_dim, mode),
                                header=False, index=False)
    pd.DataFrame(smodel_df).to_csv(
        './synthetic/logitdata_{}_{}/{}/list_impression.csv'.format(sample_num, context_dim, mode), header=False,
        index=False)
    return all_, user_feature_dict, item_feature_dict


def random_impression_list(context_dim=32, impression_len=5, sample_size=128, item_size=16, step=10, mode='train',
                           policy='random', sample_num=50, k=5):
#     x, a = user_item_feature(context_dim=context_dim, sample_size=sample_size, item_size=item_size)
#     user_feature_dict = dict(zip([j for j in range(sample_size)], list(x)))
#     item_feature_dict = dict(zip([j for j in range(item_size)], list(a)))
#     if mode == 'train':
#         x, a = user_item_feature(context_dim=context_dim, sample_size=sample_size, item_size=item_size)
#         user_feature_dict = dict(zip([j for j in range(sample_size)], list(x)))
#         item_feature_dict = dict(zip([j for j in range(item_size)], list(a)))
#         np.save('./synthetic/logitdata_{}_{}/user'.format(sample_num, context_dim), x)
#         np.save('./synthetic/logitdata_{}_{}/item'.format(sample_num, context_dim), a)
#         np.save('./synthetic/logitdata_{}_{}/user_features'.format(sample_num, context_dim), user_feature_dict)
#         np.save('./synthetic/logitdata_{}_{}/item_features'.format(sample_num, context_dim), item_feature_dict)
#     else:
    x = np.load('./synthetic/logitdata_{}_{}/user'.format(sample_num, context_dim), allow_pickle=True)
    a = np.load('./synthetic/logitdata_{}_{}/user'.format(sample_num, context_dim), allow_pickle=True)
    user_feature_dict = dict(zip([j for j in range(sample_size)], list(x)))
    item_feature_dict = dict(zip([j for j in range(item_size)], list(a)))
    # print(x)
    if not os.path.exists(
            './synthetic/randomdata_{}_{}/'.format(sample_num, context_dim)):  # 鍒ゆ柇鎵€鍦ㄧ洰褰曚笅鏄惁鏈夎鏂囦欢鍚嶇殑鏂囦欢澶?
        os.makedirs('./synthetic/randomdata_{}_{}/train/'.format(sample_num, context_dim))
        os.makedirs('./synthetic/randomdata_{}_{}/dev/'.format(sample_num, context_dim))
    np.save('./synthetic/randomdata_{}_{}/user'.format(sample_num, context_dim), x)
    np.save('./synthetic/randomdata_{}_{}/item'.format(sample_num, context_dim), a)
    np.save('./synthetic/randomdata_{}_{}/user_features'.format(sample_num, context_dim), user_feature_dict)
    np.save('./synthetic/randomdata_{}_{}/item_features'.format(sample_num, context_dim), item_feature_dict)
    all_ = None
    smodel_all = None
    for i in range(step):
        # 闅忔満閲囨牱鍑烘潵涓€缁刬tem
        # impression_list = np.random.randint(0, high = item_size, size = (sample_size, impression_len), dtype = 'l')
        impression_list = np.array(list(
            map(random_policy, [impression_len for j in range(sample_size)], [item_size for j in range(sample_size)])))
        # 鑾峰緱impression list 鐨刦eature
        # print(impression_list)
        impression_information = np.array(list(map(get_impression_feature, impression_list.reshape(-1),
                                                   [[policy, sample_num, context_dim] for j in
                                                    range(sample_size * impression_len)]))).reshape(
            [sample_size, impression_len, context_dim])
        # print(x.shape)
        # 鑾峰緱 result
        pair_matrix = np.array(list(map(nonlinear_reward_function, x, impression_information))).reshape(
            impression_list.shape)
        # print(pair_matrix.shape, impression_list.shape)
        # index = torch.topk(torch.from_numpy(pair_matrix), 1, dim=1, largest=True, sorted=True, out=None)[1].numpy()
        user_feedback = pair_matrix
        # assert index.shape[0] == user_feedback.shape[0]

        # user_feedback = np.array(list(map(get_feedbacks, user_feedback, index)))

        #         assert impression_list.shape == user_feedback.shape

        user_list = np.repeat(np.arange(sample_size).reshape([sample_size, 1]), impression_len, axis=1).reshape(-1, 1)
        impression_list = impression_list.reshape(-1, 1)
        impression_indicate = np.ones([sample_size, impression_len]).reshape(-1, 1)
        user_feedback = user_feedback.reshape(-1, 1)
        impression_id = i * sample_size + user_list
        #         assert user_list.shape == impression_list.shape

        batch = np.concatenate(
            (user_list, impression_list, impression_indicate, user_feedback, impression_id.reshape(-1, 1)), axis=1)
        #         print(batch)
        if all_ is None:
            all_ = batch
        else:
            all_ = np.concatenate((all_, batch), axis=0)
    smodel_df = save_s_model_data(all_[:, 0], all_[:, 1], all_[:, 3], impression_len, k)
    all_ = pd.DataFrame(all_)
    all_.columns = ['user', 'item', 'impression', 'click', 'impression_id']
    bpr_df = save_bpr_data(all_)

    pd.DataFrame(all_[['user', 'item', 'impression', 'click']]).to_csv(
        './synthetic/randomdata_{}_{}/{}/data.csv'.format(sample_num, context_dim, mode), header=False, index=False)
    pd.DataFrame(bpr_df).to_csv('./synthetic/randomdata_{}_{}/{}/bpr_data.csv'.format(sample_num, context_dim, mode),
                                header=False, index=False)
    pd.DataFrame(smodel_df).to_csv(
        './synthetic/randomdata_{}_{}/{}/list_impression.csv'.format(sample_num, context_dim, mode), header=False,
        index=False)
    return all_  # , user_feature_dict, item_feature_dict


# for sam in [25, 50]:
#     for cdim in [16, 32]:
#         logit_impression_list_new(mode='train', step=sam, sample_num=sam, context_dim=cdim)
#         logit_impression_list_new(mode='dev', step=10, sample_num=sam, context_dim=cdim)


#         random_impression_list(mode='train', step=sam, policy='random', sample_num=sam, context_dim=cdim)
#         random_impression_list(mode='dev', step=10, policy='random', sample_num=sam, context_dim=cdim)


# # does not work
# for sam in [5, 10, 25, 50]:
#     for cdim in [16, 32]:
#         # logit_impression_list_new(mode='train', step=sam, sample_num=sam, context_dim=cdim)
#         # logit_impression_list_new(mode='dev', step=10, sample_num=sam, context_dim=cdim)
#         # random_impression_list(mode='train', step=sam, policy='random', sample_num=sam, context_dim=cdim)
#         # random_impression_list(mode='dev', step=10, policy='random', sample_num=sam, context_dim=cdim)

#         logit_impression_list_new(context_dim=cdim, impression_len=5, sample_size=1000, item_size=100, step=sam, mode='train', sample_num=sam, k=5)
#         logit_impression_list_new(context_dim=cdim, impression_len=5, sample_size=1000, item_size=100, step=5, mode='dev', sample_num=sam, k=5)


for sam in [5]:
    for cdim in [16]:
        # logit_impression_list_new(mode='train', step=sam, sample_num=sam, context_dim=cdim)
        # logit_impression_list_new(mode='dev', step=10, sample_num=sam, context_dim=cdim)
        # random_impression_list(mode='train', step=sam, policy='random', sample_num=sam, context_dim=cdim)
        # random_impression_list(mode='dev', step=10, policy='random', sample_num=sam, context_dim=cdim)

        logit_impression_list_new(context_dim=cdim, impression_len=5, sample_size=128, item_size=16, step=sam,
                                  mode='train', sample_num=sam, k=5)
        logit_impression_list_new(context_dim=cdim, impression_len=5, sample_size=128, item_size=16, step=5, mode='dev',
                                  sample_num=sam, k=5)