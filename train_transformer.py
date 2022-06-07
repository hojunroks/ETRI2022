from model import LSTM, TransformerBased, generate_square_subsequent_mask
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from data import load_data, load_data_sequential, load_data_transformer
from argparse import ArgumentParser
from utils import evaluate
from torch.utils.tensorboard import SummaryWriter

import datetime
import pdb

PERSON_DIRS = [0,6,10,12,20,25,30]

def main():
    ########## PARSE ARUGMENTS ###########
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default="transformer")
    parser.add_argument('--data_dir', type=str, default="/data/etri_lifelog")
    parser.add_argument('--act_flag', type=str, default="act")
    parser.add_argument('--person_index', type=int, default=28)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--d_hid', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--test_every', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--split_ratio', type=float, default=0.67)
    parser.add_argument('--use_timestamp', type=bool, default=False)
    parser.add_argument('--sequence_length', type=int, default=10)
    parser.add_argument('--sche', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99)
    args = parser.parse_args()

    ############## LOAD DATA ##############
    
    person_dir_index = 0
    while person_dir_index < len(PERSON_DIRS) and PERSON_DIRS[person_dir_index] < args.person_index:
        person_dir_index += 1
    data_path="user{0:02d}-{1:02d}/user{2:02d}".format(PERSON_DIRS[person_dir_index-1]+1, PERSON_DIRS[person_dir_index], args.person_index)
    print(data_path)

    data_path = os.path.join(args.data_dir, data_path)
    train_feat, train_label, train_label_emotion, test_feat, test_label, test_label_emotion, num_classes = load_data_transformer(data_path, split_ratio=args.split_ratio, act_flag=args.act_flag, seq_len=args.sequence_length)
    with torch.cuda.device(0):
        train_feat = train_feat.cuda()
        test_feat = test_feat.cuda()
        label_act_train = train_label.cuda()
        label_act_test = test_label.cuda()
        label_emotion_train = train_label_emotion.cuda()
        label_emotion_test = test_label_emotion.cuda()

    ########### INITIALIZE MODEL ###########
    input_size = train_feat.shape[-1]
    model = TransformerBased(input_size, args.d_model, args.nhead, args.d_hid, args.nlayers, num_classes, args.dropout).to('cuda:0')
    criterion = nn.CrossEntropyLoss()
    criterion_emotion = nn.HuberLoss() # L1Loss or HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    now = datetime.datetime.now()
    nowDate = now.strftime('%Y-%m-%d-%H:%M:%S')
    if args.sche == True:
        exp_name = 'USER_%s_c%s_dh%s_dm%s_he%s_l%s_lr%s_wd%s_dr%s_g%s_s%s'%(args.person_index, num_classes, args.d_hid, args.d_model, args.nhead, args.nlayers, args.lr, args.weight_decay, args.dropout, args.gamma, args.sequence_length)
    else:
        exp_name = 'USER_%s_c%s_dh%s_dm%s_he%s_l%s_lr%s_wd%s_dr%s_nosche_s%s'%(args.person_index, num_classes, args.d_hid, args.d_model, args.nhead, args.nlayers, args.lr, args.weight_decay, args.dropout, args.sequence_length)
    writer = SummaryWriter('runs/'+exp_name)

    ################ TRAIN #################

    best_accuracy = 0.
    for epoch in range(args.num_epochs):
        model.train()
        src_mask = generate_square_subsequent_mask(args.sequence_length).to('cuda:0')
        loss_act_total = []
        loss_emo_total = []
        act_accu_total = []
        emo_accu_total = []
        
        for i in range(len(train_feat)):
            train_batch = train_feat[i].unsqueeze(dim=1)
            outputs = model(train_batch, src_mask)
            out_act, out_emotion = outputs
            optimizer.zero_grad()

            loss_act = criterion(out_act, label_act_train[i].type(torch.LongTensor).to('cuda:0'))     #trainY one-hot index
            loss_act_total.append(loss_act.item())
            loss_emo = criterion_emotion(out_emotion.squeeze(), label_emotion_train[i])
            loss_emo_total.append(loss_emo.item())
            (loss_act + loss_emo).backward() # TODO: hyperparams?
            optimizer.step()
        
            act_accu, emo_accu = evaluate(model, train_feat[i].unsqueeze(dim=1), label_act_train[i], label_emotion_train[i], args.sequence_length, args.model_name)
            act_accu_total.append(act_accu)
            emo_accu_total.append(emo_accu)

        loss_act_total = np.array(loss_act_total)
        loss_emo_total = np.array(loss_emo_total)
        act_accu_total = np.mean(np.array(act_accu_total), axis=0)
        emo_accu_total = np.mean(np.array(emo_accu_total), axis=0)
        
        writer.add_scalar("Loss/train/emo", loss_emo_total.mean(), epoch)
        writer.add_scalar("Loss/train/act", loss_act_total.mean(), epoch)
        writer.add_scalar("Accuracy/train/act-top-5", act_accu_total[2], epoch)
        writer.add_scalar("Accuracy/train/act-top-3", act_accu_total[1], epoch)
        writer.add_scalar("Accuracy/train/act-top-1", act_accu_total[0], epoch)
        writer.add_scalar("Accuracy/train/emo-top-1", emo_accu_total[0], epoch)

        if args.sche == True:
            scheduler.step()
        
        if epoch % args.test_every == 0:
            with torch.no_grad():
                src_mask = generate_square_subsequent_mask(args.sequence_length).to('cuda:0')
                model.eval()
                loss_act_test = []
                loss_emo_test = []
                act_accu_test = []
                emo_accu_test = []

                for i in range(len(test_feat)):
                    test_pred = model(test_feat[i].unsqueeze(dim=1), src_mask)
                    pred_act, pred_emotion = test_pred
                    loss_act_test.append(criterion(pred_act, label_act_test[i].type(torch.LongTensor).to('cuda:0')).item())
                    loss_emo_test.append(criterion_emotion(pred_emotion.squeeze(), label_emotion_test[i]).item())
                    
                    act_accu, emo_accu = evaluate(model, test_feat[i].unsqueeze(dim=1), label_act_test[i], label_emotion_test[i], args.sequence_length, args.model_name)
                    act_accu_test.append(act_accu)
                    emo_accu_test.append(emo_accu)
            
                act_accu_test = np.mean(np.array(act_accu_test), axis=0)
                emo_accu_test = np.mean(np.array(emo_accu_test), axis=0)
                
                loss_act_test = np.array(loss_act_test)
                loss_emo_test = np.array(loss_emo_test)
                
                writer.add_scalar("Loss/test/emo", loss_emo_test.mean(), epoch)
                writer.add_scalar("Loss/test/act", loss_act_test.mean(), epoch)
                writer.add_scalar("Accuracy/test/act-top-5", act_accu_test[2], epoch)
                writer.add_scalar("Accuracy/test/act-top-3", act_accu_test[1], epoch)
                writer.add_scalar("Accuracy/test/act-top-1", act_accu_test[0], epoch)
                writer.add_scalar("Accuracy/test/emo-top-1", emo_accu_test[0], epoch)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        
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
