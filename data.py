import os
import pandas as pd
from utils import dfToTensor, indexToOneHot
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


def load_data_sequential(data_path, split_ratio=0.67, input_flag="multi"):
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
    onehot_emotion = indexToOneHot(dfToTensor(df_sort,['emotionPositive']))[0]
    
    # emotion_label = torch.argmax(Variable(torch.Tensor(np.array(onehot_emotion[1:]))), dim=-1)
    
    data = (onehot_act, onehot_place, onehot_emotion)
    x, y, y_emotion = sliding_window(data, onehot_act[1:], onehot_emotion[1:], seq_length=3)
    
    num_data = len(x)
    train_size = int(num_data*split_ratio)
    test_size = num_data-train_size

    trainX = Variable(torch.Tensor(np.array(x[0:train_size]))).unsqueeze(dim=0)
    trainY = Variable(torch.Tensor(np.array(y[0:train_size]))).long()
    trainY_emotion = Variable(torch.Tensor(np.array(y_emotion[0:train_size]))).long()

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)]))).unsqueeze(dim=0)
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)]))).long()
    testY_emotion = Variable(torch.Tensor(np.array(y_emotion[train_size:len(y_emotion)]))).long()
    
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
        _y_emotion = torch.argmax(Variable(torch.Tensor(np.array(label_emotion[i+seq_length]))), dim=-1)
        x.append(torch.cat(seq_list).numpy())
        y.append(_y)
        y_emotion.append(_y_emotion)
        seq_list.clear()

    return np.array(x),np.array(y), np.array(y_emotion)