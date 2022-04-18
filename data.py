import os
import pandas as pd
from utils import dfToTensor, indexToOneHot
from torch.autograd import Variable
import torch
import numpy as np

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

    onehot_actopt = indexToOneHot(dfToTensor(df_sort,['actionOption']))[0]
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