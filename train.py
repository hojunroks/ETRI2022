from model import LSTM
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from data import load_data, load_data_sequential
from argparse import ArgumentParser
from utils import evaluate
from torch.utils.tensorboard import SummaryWriter

import datetime
import pdb

actionOption_dict = {'111': 'Sleep', '112': 'Sleepless', '121': 'Meal', '122': 'Snack', '131': 'Medical services, treatments, sick rest', '132': 'Personal hygiene (bath)', '133': 'Appearance management (makeup, change of clothes)', '134': 'Beauty-related services', '211': 'Main job', '212': 'Side job', '213': 'Rest during work', '22': 'Job search', '311': 'School class / seminar (listening)', '312': 'Break between classes', '313': 'School homework, self-study (individual)', '314': 'Team project (in groups)', '321': 'Private tutoring (offline)', '322': 'Online courses', '41': 'Preparing food and washing dishes', '42': 'Laundry and ironing', '43': 'Housing management and cleaning', '44': 'Vehicle management', '45': 'Pet and plant caring', '46': 'Purchasing goods and services (grocery/take-out)', '51': 'Caring for children under 10 who live together', '52': 'Caring for elementary, middle, and high school students over 10 who live together', '53': 'Caring for a spouse', '54': 'Caring for parents and grandparents who live together', '55': 'Caring for other family members who live together', '56': 'Caring for parents and grandparents who do not live together', '57': 'Caring for other family members who do not live together', '81': 'Personal care-related travel', '82': 'Commuting and work-related travel', '83': 'Education-related travel', '84': 'Travel related to housing management', '85': 'Travel related to caring for family and household members', '86': 'Travel related to participation and volunteering', '87': 'Socializing and leisure-related travel', '61': 'Religious activities', '62': 'Political activity', '63': 'Ceremonial activities', '64': 'Volunteer', '711': 'Offline communication', '712': 'Video or voice call', '713': 'Text or email (Online)', '721': 'Reading books, newspapers, and magazines', '722': 'Watching TV or video', '723': 'Listening to audio', '724': 'Internet search or blogging', '725': 'Gaming (mobile, computer, video)', '741': 'Watching a sporting event', '742': 'Watching movie', '743': 'Concerts and plays', '744': 'Amusement Park, zoo', '745': 'Festival, carnival', '746': 'Driving, sightseeing, excursion', '751': 'Walking', '752': 'Running, jogging', '753': 'Climbing, hiking', '754': 'Biking', '755': 'Ball games (soccer, basketball, baseball, tennis, etc)', '756': 'Camping, fishing', '761': 'Group games (board games, card games, puzzles, etc.)', '762': 'Personal hobbies (woodworking, gardening, etc.)', '763': 'Group performances (orchestra, choir, troupe, etc.)', '764': 'Liberal arts and learning (languages, musical instruments, etc.)', '791': 'Nightlife', '792': 'Smoking', '793': 'Do nothing and rest', '91': 'Online shopping', '92': 'Offline shopping'}
PERSON_DIRS = [0,6,10,12,20,25,30]

def main():
    ########## PARSE ARUGMENTS ###########
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/data/etri_lifelog")
    parser.add_argument('--person_index', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--test_every', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--split_ratio', type=float, default=0.67)
    parser.add_argument('--use_timestamp', type=bool, default=False)
    args = parser.parse_args()

    ############## LOAD DATA ##############
    
    person_dir_index = 0
    while person_dir_index < len(PERSON_DIRS) and PERSON_DIRS[person_dir_index] < args.person_index:
        person_dir_index += 1
    data_path="user{0:02d}-{1:02d}/user{2:02d}".format(PERSON_DIRS[person_dir_index-1]+1, PERSON_DIRS[person_dir_index], args.person_index)
    print(data_path)
    # train_feat, train_label, test_feat, test_label = load_data(os.path.join(args.data_dir, data_path), split_ratio=args.split_ratio, input_flag="multi")  # act_place_emo
    data_path = os.path.join(args.data_dir, data_path)
    train_feat, train_label, train_label_emotion, test_feat, test_label, test_label_emotion = load_data_sequential(data_path, split_ratio=args.split_ratio)
    with torch.cuda.device(0):
        train_feat = train_feat.cuda()
        test_feat = test_feat.cuda()
        label_act_train = train_label.cuda() # label_emotion_train
        label_act_test = test_label.cuda() # label_emotion_test
        label_emotion_train = train_label_emotion.cuda()
        label_emotion_test = test_label_emotion.cuda()

    ########### INITIALIZE MODEL ###########
    input_size = train_feat.shape[-1]
    num_classes = 16    # train_feat.shape[-1]
    lstm = LSTM(num_classes, input_size, args.hidden_size, args.num_layers, bidirectional_flag=args.bidirectional, dropout=args.dropout).to(0)
    #classfication
    criterion = nn.CrossEntropyLoss()
    criterion_emotion = nn.HuberLoss() # L1Loss or HuberLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    now = datetime.datetime.now()
    nowDate = now.strftime('%Y-%m-%d-%H:%M:%S')
    writer = SummaryWriter('runs/USER_{}_{}'.format(args.person_index, nowDate))


    ################ TRAIN #################

    best_accuracy = 0.
    # Train the model
    for epoch in range(args.num_epochs):
        lstm.train()
        
        outputs = lstm(train_feat)
        out_act, out_emotion = outputs
        
        optimizer.zero_grad()
        
        loss_act = criterion(out_act, label_act_train)     #trainY one-hot index
        
        loss_emotion = criterion_emotion(out_emotion.squeeze(), label_emotion_train)
        (loss_act + loss_emotion).backward()
        optimizer.step()
        
        accuracy = np.array(evaluate(lstm, train_feat, label_act_train))
        
        # Logs
        writer.add_scalar("Loss/train", loss_act, epoch)

        writer.add_scalar("Accuracy/train/top-5", accuracy[2], epoch)
        writer.add_scalar("Accuracy/train/top-3", accuracy[1], epoch)
        writer.add_scalar("Accuracy/train/top-1", accuracy[0], epoch)
        writer.add_scalar("Accuracy/train/emotion", loss_emotion, epoch)
        
        if epoch % args.test_every == 0:
            with torch.no_grad():
                lstm.eval()
                test_pred = lstm(test_feat)
                pred_act, pred_emotion = test_pred
                loss_act_test = criterion(pred_act, label_act_test)
                loss_emotion_test = criterion_emotion(pred_emotion.squeeze(), label_emotion_test)
                accuracy_test = np.array(evaluate(lstm, test_feat, label_act_test))
                writer.add_scalar("Loss/test", loss_act_test, epoch)
                writer.add_scalar("Accuracy/test/top-5", accuracy_test[2], epoch)
                writer.add_scalar("Accuracy/test/top-3", accuracy_test[1], epoch)
                writer.add_scalar("Accuracy/test/top-1", accuracy_test[0], epoch)
                
                writer.add_scalar("Accuracy/test/emotion", loss_emotion_test, epoch)
        
            if best_accuracy <= accuracy[0]:
                best_accuracy = accuracy[0]
                save_dict = {
                        "epoch": epoch,
                        "model_state_dict": lstm.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "accuracy": best_accuracy,
                    }
                save_path = "./wgt/user" + str(args.person_index) + "_" + str(epoch) + ".ckpt"
                torch.save(save_dict, save_path)                   


if __name__=='__main__':
    main()
