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

    
    num_data = len(x)
    train_size = int(num_data*split_ratio)
    test_size = num_data-train_size

    trainX = Variable(torch.Tensor(np.array(x[0:train_size]))).unsqueeze(dim=0)
    trainY = Variable(torch.Tensor(np.array(y[0:train_size]))).long()
    trainY_emotion = Variable(torch.Tensor(np.array(y_emotion[0:train_size]))).float()

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)]))).unsqueeze(dim=0)
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)]))).long()
    testY_emotion = Variable(torch.Tensor(np.array(y_emotion[train_size:len(y_emotion)]))).float()
    
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

    return np.array(x),np.array(y), np.array(y_emotion)


# def load_data_transformer(data_path, split_ratio=0.67, batch_size=64, use_timestamp=True):
#     dir_list = os.listdir(data_path)
#     user_all_data = []

#     for item in dir_list:
#         csv_file = data_path + "/" + item + "/" + item + "_label.csv"
#         rdr = pd.read_csv(csv_file)
#         user_all_data.append(rdr)


#     #df_all : 전체 data
#     df_all = pd.concat(user_all_data, axis=0)
#     df_all['ts']=pd.to_datetime(df_all['ts'], unit='s')
#     df_all['is_weekend']=df_all.apply(lambda x: x['ts'].weekday()>=5, axis=1)
#     df_all['ampm']=df_all.apply(lambda x: ["AM","PM"][x['ts'].hour//12], axis=1)
#     df_all['time']=df_all.apply(lambda x: hourToLabel(x['ts'].hour), axis=1)

#     df_no_dup = df_all.drop_duplicates()
#     #df_sort : 연속된 동일한 actionOption끼리 묶음
#     diff_indices = [0]
#     diff_index = 0
#     df_sort = pd.DataFrame()
#     for i in range(len(df_no_dup)):
#         if(df_no_dup['actionOption'].values[i]!=df_no_dup['actionOption'].values[diff_index]):
#             diff_index=i
#             diff_indices.append(diff_index)
#             df_sort = df_sort.append(pd.Series(df_no_dup.iloc[i], index=df_no_dup.columns), ignore_index=True)

#     # onehot_actopt = indexToOneHot(dfToTensor(df_sort,['actionOption']))[0]
#     onehot_act = indexToOneHot(dfToTensor(df_sort,['action']))[0]
#     onehot_place = indexToOneHot(dfToTensor(df_sort,['place']))[0]
#     raw_emotion = torch.from_numpy(np.array(df_sort['emotionPositive'] / 7.)).unsqueeze(dim=1)
#     onehot_weekend = torch.unsqueeze(dfToTensor(df_sort,['is_weekend'])[0], dim=1)
#     # onehot_time = indexToOneHot(dfToTensor(df_sort,['time']))[0]
#     time = torch.unsqueeze(torch.Tensor([(x.hour/12-1) for x in list(df_sort['ts'])]), dim=1)
#     # emotion_label = torch.argmax(Variable(torch.Tensor(np.array(onehot_emotion[1:]))), dim=-1)
    
#     if use_timestamp:
#         data = (onehot_act, onehot_place, raw_emotion, onehot_weekend, time)
#     else:
#         data = (onehot_act, onehot_place, raw_emotion)
#     data = batchify(data, batch_size) # N/bsz, bsz, input_size
    


    
#     num_data = len(x)
#     train_size = int(num_data*split_ratio)
#     test_size = num_data-train_size

#     trainX = Variable(torch.Tensor(np.array(x[0:train_size]))).unsqueeze(dim=0)
#     trainY = Variable(torch.Tensor(np.array(y[0:train_size]))).long()
#     trainY_emotion = Variable(torch.Tensor(np.array(y_emotion[0:train_size]))).float()

#     testX = Variable(torch.Tensor(np.array(x[train_size:len(x)]))).unsqueeze(dim=0)
#     testY = Variable(torch.Tensor(np.array(y[train_size:len(y)]))).long()
#     testY_emotion = Variable(torch.Tensor(np.array(y_emotion[train_size:len(y_emotion)]))).float()
    
#     return trainX, trainY, trainY_emotion, testX, testY, testY_emotion
    


# def batchify(data, bsz):
#     """Divides the data into bsz separate sequences, removing extra elements
#     that wouldn't cleanly fit.

#     Args:
#         data: Tensor, shape [N,input_size]
#         bsz: int, batch size

#     Returns:
#         Tensor of shape [N // bsz, bsz]
#     """
#     seq_len = data.size[0] // bsz
#     data = data[:seq_len * bsz]
#     data = data.view(bsz, seq_len, -1).transpose(0,1).contiguous()
#     return data

# bptt = 35
# def get_batch(source, i):
#     """
#     Args:
#         source: Tensor, shape [full_seq_len, batch_size]
#         i: int

#     Returns:
#         tuple (data, target), where data has shape [seq_len, batch_size] and
#         target has shape [seq_len * batch_size]
#     """
#     seq_len = min(bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].reshape(-1)
#     return data, target