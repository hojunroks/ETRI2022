from model import LSTM, TransformerBased, MLP
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from data import load_data, load_data_lstm, load_data_mlp
from argparse import ArgumentParser
from utils import evaluate
from torch.utils.tensorboard import SummaryWriter

import datetime
import pdb

PERSON_DIRS = [0,6,10,12,20,25,30]

def main():
    ########## PARSE ARUGMENTS ###########
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default="lstm")
    parser.add_argument('--data_dir', type=str, default="/data/etri_lifelog")
    parser.add_argument('--act_flag', type=str, default="act")
    parser.add_argument('--person_index', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--test_every', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--split_ratio', type=float, default=0.67)
    parser.add_argument('--use_timestamp', type=bool, default=False)
    parser.add_argument('--sequence_length', type=int, default=10)
    args = parser.parse_args()

    ############## LOAD DATA ##############
    
    person_dir_index = 0
    while person_dir_index < len(PERSON_DIRS) and PERSON_DIRS[person_dir_index] < args.person_index:
        person_dir_index += 1
    data_path="user{0:02d}-{1:02d}/user{2:02d}".format(PERSON_DIRS[person_dir_index-1]+1, PERSON_DIRS[person_dir_index], args.person_index)
    print(data_path)

    data_path = os.path.join(args.data_dir, data_path)
    if args.model_name=='lstm':
        train_feat, train_label, train_label_emotion, test_feat, test_label, test_label_emotion, num_classes = load_data_lstm(data_path, split_ratio=args.split_ratio, act_flag=args.act_flag, seq_len=args.sequence_length)
    elif args.model_name=='MLP':
        train_feat, train_label, train_label_emotion, test_feat, test_label, test_label_emotion, num_classes = load_data_mlp(data_path, split_ratio=args.split_ratio, act_flag=args.act_flag)
   
    with torch.cuda.device(0):
        train_feat = train_feat.cuda()
        test_feat = test_feat.cuda()
        label_act_train = train_label.cuda() # label_emotion_train
        label_act_test = test_label.cuda() # label_emotion_test
        label_emotion_train = train_label_emotion.cuda()
        label_emotion_test = test_label_emotion.cuda()

    ########### INITIALIZE MODEL ###########
    input_size = train_feat.shape[-1]
    if args.model_name=='lstm':
        model = LSTM(num_classes, input_size, args.hidden_size, args.num_layers, bidirectional_flag=args.bidirectional, dropout=args.dropout).to(0)
    elif args.model_name=='MLP':
        model = MLP(num_classes, input_size, args.hidden_size, args.num_layers, args.dropout).to(0)

    criterion = nn.CrossEntropyLoss()
    criterion_emotion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    now = datetime.datetime.now()
    nowDate = now.strftime('%Y-%m-%d-%H:%M:%S')
    if args.model_name == 'lstm':
        exp_name = 'LSTM_USER_%s_c%s_lr%s_wd%s_dr%s_bi_s%s'%(args.person_index, num_classes, args.lr, args.weight_decay, args.dropout, args.sequence_length)
    elif args.model_name == 'mlp':
        exp_name = 'MLP_USER_%s_c%s_lr%s_wd%s_dr%s_bi_s%s'%(args.person_index, num_classes, args.lr, args.weight_decay, args.dropout, args.sequence_length)
    writer = SummaryWriter('runs/'+exp_name)
    
    ################ TRAIN #################

    best_accuracy = 0.
    for epoch in range(args.num_epochs):
        model.train()
        
        outputs = model(train_feat)
        out_act, out_emotion = outputs
        optimizer.zero_grad()
        loss_act = criterion(out_act, label_act_train)     #trainY one-hot index
        loss_emo = criterion_emotion(out_emotion.squeeze(), label_emotion_train)
        (loss_act + loss_emo).backward()
        optimizer.step()
        
        act_accu, emo_accu = evaluate(model, train_feat, label_act_train, label_emotion_train, args.sequence_length, args.model_name)
        
        writer.add_scalar("Loss/train/emo", loss_emo, epoch)
        writer.add_scalar("Loss/train/act", loss_act, epoch)
        writer.add_scalar("Accuracy/train/act-top-5", act_accu[2], epoch)
        writer.add_scalar("Accuracy/train/act-top-3", act_accu[1], epoch)
        writer.add_scalar("Accuracy/train/act-top-1", act_accu[0], epoch)
        writer.add_scalar("Accuracy/train/emo-top-1", emo_accu[0], epoch)
        
        # print("Test")
        if epoch % args.test_every == 0:
            with torch.no_grad():
                model.eval()
                test_pred = model(test_feat)
                pred_act, pred_emotion = test_pred
                loss_act_test = criterion(pred_act, label_act_test)
                loss_emo_test = criterion_emotion(pred_emotion.squeeze(), label_emotion_test)
                
                act_accu_test, emo_accu_test = evaluate(model, test_feat, label_act_test, label_emotion_test, args.sequence_length, args.model_name)

                writer.add_scalar("Loss/test/emo", loss_emo_test, epoch)
                writer.add_scalar("Loss/test/act", loss_act_test, epoch)
                writer.add_scalar("Accuracy/test/act-top-5", act_accu_test[2], epoch)
                writer.add_scalar("Accuracy/test/act-top-3", act_accu_test[1], epoch)
                writer.add_scalar("Accuracy/test/act-top-1", act_accu_test[0], epoch)
                writer.add_scalar("Accuracy/test/emo-top-1", emo_accu_test[0], epoch)
                
            if best_accuracy <= act_accu_test[0]: # based on act top-1 accuracy
                best_accuracy = act_accu_test[0]
                best_accus = np.concatenate((act_accu_test, emo_accu_test), axis=-1)
                
                save_dict = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "accuracy": best_accuracy,
                    }
                
                save_path = "./wgt/" + exp_name
                if os.path.exists(save_path) == False:
                    os.mkdir(save_path)
                    
                save_model_name = os.path.join(save_path, 'best.ckpt')
                torch.save(save_dict, save_model_name) 
                np.save(os.path.join(save_path, "best_accu.npy"), best_accus)
                    
if __name__=='__main__':
    main()
