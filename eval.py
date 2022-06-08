from model import LSTM
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from data import load_data, load_data_sequential
from argparse import ArgumentParser
from utils import evaluate

import pdb
from baal.active.heuristics import BALD, Entropy, Variance, Margin

actionOption_dict = {'111': 'Sleep', '112': 'Sleepless', '121': 'Meal', '122': 'Snack', '131': 'Medical services, treatments, sick rest', '132': 'Personal hygiene (bath)', '133': 'Appearance management (makeup, change of clothes)', '134': 'Beauty-related services', '211': 'Main job', '212': 'Side job', '213': 'Rest during work', '22': 'Job search', '311': 'School class / seminar (listening)', '312': 'Break between classes', '313': 'School homework, self-study (individual)', '314': 'Team project (in groups)', '321': 'Private tutoring (offline)', '322': 'Online courses', '41': 'Preparing food and washing dishes', '42': 'Laundry and ironing', '43': 'Housing management and cleaning', '44': 'Vehicle management', '45': 'Pet and plant caring', '46': 'Purchasing goods and services (grocery/take-out)', '51': 'Caring for children under 10 who live together', '52': 'Caring for elementary, middle, and high school students over 10 who live together', '53': 'Caring for a spouse', '54': 'Caring for parents and grandparents who live together', '55': 'Caring for other family members who live together', '56': 'Caring for parents and grandparents who do not live together', '57': 'Caring for other family members who do not live together', '81': 'Personal care-related travel', '82': 'Commuting and work-related travel', '83': 'Education-related travel', '84': 'Travel related to housing management', '85': 'Travel related to caring for family and household members', '86': 'Travel related to participation and volunteering', '87': 'Socializing and leisure-related travel', '61': 'Religious activities', '62': 'Political activity', '63': 'Ceremonial activities', '64': 'Volunteer', '711': 'Offline communication', '712': 'Video or voice call', '713': 'Text or email (Online)', '721': 'Reading books, newspapers, and magazines', '722': 'Watching TV or video', '723': 'Listening to audio', '724': 'Internet search or blogging', '725': 'Gaming (mobile, computer, video)', '741': 'Watching a sporting event', '742': 'Watching movie', '743': 'Concerts and plays', '744': 'Amusement Park, zoo', '745': 'Festival, carnival', '746': 'Driving, sightseeing, excursion', '751': 'Walking', '752': 'Running, jogging', '753': 'Climbing, hiking', '754': 'Biking', '755': 'Ball games (soccer, basketball, baseball, tennis, etc)', '756': 'Camping, fishing', '761': 'Group games (board games, card games, puzzles, etc.)', '762': 'Personal hobbies (woodworking, gardening, etc.)', '763': 'Group performances (orchestra, choir, troupe, etc.)', '764': 'Liberal arts and learning (languages, musical instruments, etc.)', '791': 'Nightlife', '792': 'Smoking', '793': 'Do nothing and rest', '91': 'Online shopping', '92': 'Offline shopping'}
PERSON_DIRS = [0,6,10,12,20,25,30]

def main():
    ########## PARSE ARUGMENTS ###########
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/data/etri_lifelog")
    parser.add_argument('--person_index', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--test_every', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--split_ratio', type=float, default=0.67)
    parser.add_argument('--test_epoch', type=int, default=2000)
    args = parser.parse_args()

    ############## LOAD DATA #########s#####
    
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
    num_classes = 16
    lstm = LSTM(num_classes, input_size, args.hidden_size, args.num_layers, bidirectional_flag=args.bidirectional, dropout=args.dropout).to(0)
    
    ## TODO (optional) : load multiple models
    checkpoint = torch.load("./wgt/user%s_%s.ckpt"%(args.person_index, args.test_epoch))
    
    lstm.load_state_dict(checkpoint["model_state_dict"])
    # classfication
    criterion = nn.CrossEntropyLoss()
    criterion_emotion = nn.HuberLoss() # L1Loss or HuberLoss()
    # optimizer = torch.optim.Adam(lstm.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    best_accuracy = 0.
    # Train the model
    f = open("USER%s.txt"%(args.person_index), 'w')

    with torch.no_grad():
        # lstm.eval()
        test_pred = lstm(test_feat)
        pred_act, pred_emotion = test_pred

        ## uncertainty estimation by yjheo
        # pred_act: [n_instances, n_classes] > [n_instances, n_classes, 1]   
        pred_act_prob = nn.Softmax(dim=-1)(pred_act)
        pred_act_prob = pred_act_prob.unsqueeze(-1).cpu().numpy()
        print('Scores using entropy :', Entropy().compute_score(pred_act_prob))
        print('Ranks using entropy :', Entropy()(pred_act_prob))
        print('Scores using margin :', Margin().compute_score(pred_act_prob))
        print('Ranks using margin :', Margin()(pred_act_prob))

        ## bald:
        # All uncertainty scores all zeros due to number_of_iterations=1
        bald_scores = BALD().compute_score(pred_act_prob)
        if np.sum(bald_scores) != 0:
            print('Scores using bald :', bald_scores) 
            print('Ranks using bald :', BALD()(pred_act_prob))
        
        ## variance:
        # All uncertainty scores all zeros.
        var_scores = Variance().compute_score(pred_act_prob)
        if np.sum(var_scores) != 0:
            print('Scores using variance :', var_scores)
            print('Ranks using variance :', Variance()(pred_act_prob))
        
        loss_act_test = criterion(pred_act, label_act_test)
        loss_emotion_test = criterion_emotion(pred_emotion.squeeze(), label_emotion_test/7.)
        accuracy_test = np.array(evaluate(lstm, test_feat, label_act_test))
        
        for i in range(len(pred_act)):
            print("action pred:")
            f.write("\naction pred:")
            print(pred_act[i].cpu().numpy())
            f.write(str(pred_act[i].cpu().numpy())+"\n")
            print("action label: "+str(label_act_test[i].cpu().numpy()))
            f.write("action label: "+str(label_act_test[i].cpu().numpy())+"\n")
            print("emotion pred: "+ str(pred_emotion[i].cpu().numpy()[0]))
            f.write("emotion pred: "+ str(pred_emotion[i].cpu().numpy()[0])+"\n")
            print("emotion label: "+str(label_emotion_test[i].cpu().numpy()))
            f.write("emotion label: "+str(label_emotion_test[i].cpu().numpy())+"\n")
            
            print("\n")
        
        print(accuracy_test[1])
        f.write(str(accuracy_test[1])+"% \n")
        
        f.close()

        # writer.add_scalar("Loss/test", loss_act_test, epoch)
        # writer.add_scalar("Accuracy/test/top-10", accuracy_test[2], epoch)
        # writer.add_scalar("Accuracy/test/top-5", accuracy_test[1], epoch)
        # writer.add_scalar("Accuracy/test/top-1", accuracy_test[0], epoch)
        
        # writer.add_scalar("Accuracy/test/emotion", loss_emotion_test, epoch)

if __name__=='__main__':
    main()