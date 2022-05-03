import os
import pandas as pd
from utils import dfToTensor, indexToOneHot, hourToLabel
from torch.autograd import Variable
import torch
import numpy as np

import pdb

def load_data(data_path, split_ratio=0.67, input_flag="multi"):
    dir_list = os.listdir(data_path)
    user_all_data = []

    for item in dir_list:
        csv_file = data_path + "/" + item + "/" + item + "_label.csv"
        rdr = pd.read_csv(csv_file)
        user_all_data.append(rdr)


    #df_all : 전체 data
    df_all = pd.concat(user_all_data, axis=0)
    df_all['ts']=pd.to_datetime(df_all['ts'], unit='s')
    df_all['is_weekend']=df_all.apply(lambda x: x['ts'].weekday()>=5, axis=1)
    df_all['ampm']=df_all.apply(lambda x: ["AM","PM"][x['ts'].hour//12], axis=1)
    df_all['time']=df_all.apply(lambda x: hourToLabel(x['ts'].hour), axis=1)

    df_no_dup = df_all.drop_duplicates()
    #df_sort : 연속된 동일한 actionOption끼리 묶음
    diff_indices = [0]
    diff_index = 0
    df_sort = pd.DataFrame()
    for i in range(len(df_no_dup)):
        if(df_no_dup['actionOption'].values[i]!=df_no_dup['actionOption'].values[diff_index]):
            diff_index=i
            diff_indices.append(diff_index)
            df_sort = df_sort.append(pd.Series(df_no_dup.iloc[i], index=df_no_dup.columns), ignore_index=True)

    # onehot_actopt = indexToOneHot(dfToTensor(df_sort,['actionOption']))[0]
    onehot_actopt = indexToOneHot(dfToTensor(df_sort,['action']))[0]
    onehot_place = indexToOneHot(dfToTensor(df_sort,['place']))[0]
    onehot_emotion = indexToOneHot(dfToTensor(df_sort,['emotionPositive']))[0]
    onehot_weekend = indexToOneHot(dfToTensor(df_sort,['is_weekend']))[0]
    onehot_time = indexToOneHot(dfToTensor(df_sort,['time']))[0]

    num_data = len(onehot_actopt)
    train_size = int(num_data*split_ratio)
    test_size = num_data-train_size

    actopt_feat_train = Variable(torch.Tensor(np.array(onehot_actopt[0:train_size-1])))
    actopt_feat_test = Variable(torch.Tensor(np.array(onehot_actopt[train_size:])))
    actopt_label_train = torch.argmax(Variable(torch.Tensor(np.array(onehot_actopt[1:train_size]))), dim=-1)
    actopt_label_test = torch.argmax(Variable(torch.Tensor(np.array(onehot_actopt[train_size:]))), dim=-1)

    place_feat_train = Variable(torch.Tensor(np.array(onehot_place[0:train_size-1])))
    place_feat_test = Variable(torch.Tensor(np.array(onehot_place[train_size:])))
    place_label_train = torch.argmax(Variable(torch.Tensor(np.array(onehot_place[1:train_size]))), dim=-1)
    place_label_test = torch.argmax(Variable(torch.Tensor(np.array(onehot_place[train_size:]))), dim=-1)

    emotion_feat_train = Variable(torch.Tensor(np.array(onehot_emotion[0:train_size-1])))
    emotion_feat_test = Variable(torch.Tensor(np.array(onehot_emotion[train_size:])))
    emotion_label_train = torch.argmax(Variable(torch.Tensor(np.array(onehot_emotion[1:train_size]))), dim=-1)
    emotion_label_test = torch.argmax(Variable(torch.Tensor(np.array(onehot_emotion[train_size:]))), dim=-1)

    if input_flag == "act_only": # input: actopt
        train_feat = torch.unsqueeze(actopt_feat_train, 1)
        test_feat = torch.unsqueeze(actopt_feat_test, 1)
        train_label = actopt_label_train    
        test_label = actopt_label_test
    elif input_flag == "act_place": # input: concat(actopt, place)
        train_feat = torch.unsqueeze(torch.cat((actopt_feat_train, place_feat_train), axis=-1), 1)
        test_feat = torch.unsqueeze(torch.cat((actopt_feat_test, place_feat_test), axis=-1), 1)
        train_label = actopt_label_train
        test_label = actopt_label_test
    elif input_flag == "act_place_emo": # input: concat(actopt, place, emotion)
        train_feat = torch.unsqueeze(torch.cat((actopt_feat_train, place_feat_train, emotion_feat_train), axis=-1), 1)
        test_feat = torch.unsqueeze(torch.cat((actopt_feat_test, place_feat_test, emotion_feat_test), axis=-1), 1)
        train_label = actopt_label_train
        test_label = actopt_label_test
    elif input_flag == "multi":
        train_feat = torch.unsqueeze(torch.cat((actopt_feat_train, place_feat_train, emotion_feat_train), axis=-1), 1)
        test_feat = torch.unsqueeze(torch.cat((actopt_feat_test, place_feat_test, emotion_feat_test), axis=-1), 1)
        train_label = actopt_label_train, emotion_label_train
        test_label = actopt_label_test, emotion_label_test
        
    print (train_feat.shape, actopt_label_train.shape, test_feat.shape, actopt_label_test.shape)

    return train_feat, train_label, test_feat, test_label


def load_data_sequential(data_path, split_ratio=0.67, input_flag="multi", use_timestamp=True):
    dir_list = os.listdir(data_path)
    user_all_data = []

    for item in dir_list:
        csv_file = data_path + "/" + item + "/" + item + "_label.csv"
        rdr = pd.read_csv(csv_file)
        user_all_data.append(rdr)


    #df_all : 전체 data
    df_all = pd.concat(user_all_data, axis=0)
    df_all['ts']=pd.to_datetime(df_all['ts'], unit='s')
    df_all['is_weekend']=df_all.apply(lambda x: x['ts'].weekday()>=5, axis=1)
    df_all['ampm']=df_all.apply(lambda x: ["AM","PM"][x['ts'].hour//12], axis=1)
    df_all['time']=df_all.apply(lambda x: hourToLabel(x['ts'].hour), axis=1)

    df_no_dup = df_all.drop_duplicates()
    #df_sort : 연속된 동일한 actionOption끼리 묶음
    diff_indices = [0]
    diff_index = 0
    df_sort = pd.DataFrame()
    for i in range(len(df_no_dup)):
        if(df_no_dup['actionOption'].values[i]!=df_no_dup['actionOption'].values[diff_index]):
            diff_index=i
            diff_indices.append(diff_index)
            df_sort = df_sort.append(pd.Series(df_no_dup.iloc[i], index=df_no_dup.columns), ignore_index=True)

    # onehot_actopt = indexToOneHot(dfToTensor(df_sort,['actionOption']))[0]
    onehot_act = indexToOneHot(dfToTensor(df_sort,['action']))[0]
    onehot_place = indexToOneHot(dfToTensor(df_sort,['place']))[0]
    raw_emotion = torch.from_numpy(np.array(df_sort['emotionPositive'] / 7.)).unsqueeze(dim=1)
    onehot_weekend = torch.unsqueeze(dfToTensor(df_sort,['is_weekend'])[0], dim=1)
    # onehot_time = indexToOneHot(dfToTensor(df_sort,['time']))[0]
    time = torch.unsqueeze(torch.Tensor([(x.hour/12-1) for x in list(df_sort['ts'])]), dim=1)
    # emotion_label = torch.argmax(Variable(torch.Tensor(np.array(onehot_emotion[1:]))), dim=-1)
    
    if use_timestamp:
        data = (onehot_act, onehot_place, raw_emotion, onehot_weekend, time)
    else:
        data = (onehot_act, onehot_place, raw_emotion)
    x, y, y_emotion = sliding_window(data, onehot_act[1:], raw_emotion[1:], seq_length=3)
    
    print(x.shape)


    
    num_data = len(x)
    train_size = int(num_data*split_ratio)
    test_size = num_data-train_size

    trainX = Variable(torch.Tensor(np.array(x[0:train_size]))).unsqueeze(dim=0)
    trainY = Variable(torch.Tensor(np.array(y[0:train_size]))).long()
    trainY_emotion = Variable(torch.Tensor(np.array(y_emotion[0:train_size]))).float()

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)]))).unsqueeze(dim=0)
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)]))).long()
    testY_emotion = Variable(torch.Tensor(np.array(y_emotion[train_size:len(y_emotion)]))).float()
    
    print(trainX.shape)
    print(trainY.shape)
    print(trainY_emotion.shape)

    return trainX, trainY, trainY_emotion, testX, testY, testY_emotion
    

def sliding_window(data, label, label_emotion, seq_length):
    x = []
    y = []
    y_emotion = []
    seq_list = []

    for i in range(len(data[0])-seq_length-1):
        for k in range(seq_length):
            seq_list += [data[j][i+k] for j in range(len(data))]
            
        _y = torch.argmax(Variable(torch.Tensor(np.array(label[i+seq_length]))), dim=-1)
        _y_emotion = Variable(torch.Tensor(np.array(label_emotion[i+seq_length])))
        x.append(torch.cat(seq_list).numpy())
        y.append(_y)
        y_emotion.append(_y_emotion)
        seq_list.clear()

    return np.array(x),np.array(y), np.array([y.item() for y in y_emotion])


def load_data_transformer(data_path, split_ratio=0.67, use_timestamp=True, seq_len=10):
    dir_list = os.listdir(data_path)
    user_all_data = []

    for item in dir_list:
        csv_file = data_path + "/" + item + "/" + item + "_label.csv"
        rdr = pd.read_csv(csv_file)
        user_all_data.append(rdr)


    #df_all : 전체 data
    df_all = pd.concat(user_all_data, axis=0)
    df_all['ts']=pd.to_datetime(df_all['ts'], unit='s')
    df_all['is_weekend']=df_all.apply(lambda x: x['ts'].weekday()>=5, axis=1)
    df_all['ampm']=df_all.apply(lambda x: ["AM","PM"][x['ts'].hour//12], axis=1)
    df_all['time']=df_all.apply(lambda x: hourToLabel(x['ts'].hour), axis=1)

    df_no_dup = df_all.drop_duplicates()
    #df_sort : 연속된 동일한 actionOption끼리 묶음
    diff_indices = [0]
    diff_index = 0
    df_sort = pd.DataFrame()
    for i in range(len(df_no_dup)):
        if(df_no_dup['actionOption'].values[i]!=df_no_dup['actionOption'].values[diff_index]):
            diff_index=i
            diff_indices.append(diff_index)
            df_sort = df_sort.append(pd.Series(df_no_dup.iloc[i], index=df_no_dup.columns), ignore_index=True)

    # onehot_actopt = indexToOneHot(dfToTensor(df_sort,['actionOption']))[0]
    onehot_act = indexToOneHot(dfToTensor(df_sort,['action']))[0]
    onehot_place = indexToOneHot(dfToTensor(df_sort,['place']))[0]
    raw_action = dfToTensor(df_sort,['action'])[0].unsqueeze(dim=1)
    raw_emotion = torch.from_numpy(np.array(df_sort['emotionPositive'] / 7.)).unsqueeze(dim=1)
    onehot_weekend = torch.unsqueeze(dfToTensor(df_sort,['is_weekend'])[0], dim=1)
    # onehot_time = indexToOneHot(dfToTensor(df_sort,['time']))[0]
    time = torch.unsqueeze(torch.Tensor([(x.hour/12-1) for x in list(df_sort['ts'])]), dim=1)
    # emotion_label = torch.argmax(Variable(torch.Tensor(np.array(onehot_emotion[1:]))), dim=-1)
    
    num_data = len(onehot_act)
    train_size = int(num_data*split_ratio)

    if use_timestamp:
        data_train = torch.cat([onehot_act[:train_size], onehot_place[:train_size], raw_emotion[:train_size], onehot_weekend[:train_size], time[:train_size]], dim=1)
        data_test = torch.cat([onehot_act[train_size:], onehot_place[train_size:], raw_emotion[train_size:], onehot_weekend[train_size:], time[train_size:]], dim=1)
    else:
        data_train = torch.cat([onehot_act[:train_size], onehot_place[:train_size], raw_emotion[:train_size]], dim=1)
        data_test = torch.cat([onehot_act[train_size:], onehot_place[train_size:], raw_emotion[train_size:]], dim=1)

    

    trainX, trainY, trainY_emotion = get_batch(data_train, raw_action, raw_emotion, seq_len)
    
    testX, testY, testY_emotion = get_batch(data_test, raw_action, raw_emotion, seq_len)

    # testX = Variable(torch.Tensor(np.array(x[train_size:len(x)]))).unsqueeze(dim=0)
    # testY = Variable(torch.Tensor(np.array(y[train_size:len(y)]))).long()
    # testY_emotion = Variable(torch.Tensor(np.array(y_emotion[train_size:len(y_emotion)]))).float()
    
    return trainX, trainY, trainY_emotion, testX, testY, testY_emotion
    

def get_batch(data, action_label, emotion_label, bptt):
    """
    Args:
        source: Tensor, shape [full_seq_len, input_size]

    Returns:
        tuple (data, target), where data has shape [seq_len, input_size] and
        target has shape [seq_len * input_size]
    """
    num_batches = len(data)-1-bptt
    seq_len = bptt
    x = Variable(torch.Tensor(num_batches, seq_len, data.shape[1]))
    action = Variable(torch.Tensor(num_batches, seq_len))
    emotion = Variable(torch.Tensor(num_batches, seq_len))
    for i in range(num_batches):
        x[i] = data[i:i+seq_len]
        action[i] = action_label[i+1:i+1+seq_len].squeeze()
        emotion[i] = emotion_label[i+1:i+1+seq_len].squeeze()
    
    return x, action, emotion