from model import LSTM, TransformerBased, MLP, generate_square_subsequent_mask
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from data import load_data, load_data_lstm, load_data_mlp, load_data_transformer
from argparse import ArgumentParser
from utils import evaluate, top_k, fix_seed

import pdb
import csv
from baal.active.heuristics import BALD, Entropy, Variance, Margin

actionOption_dict = {'111': 'Sleep', '112': 'Sleepless', '121': 'Meal', '122': 'Snack', '131': 'Medical services, treatments, sick rest', '132': 'Personal hygiene (bath)', '133': 'Appearance management (makeup, change of clothes)', '134': 'Beauty-related services', '211': 'Main job', '212': 'Side job', '213': 'Rest during work', '22': 'Job search', '311': 'School class / seminar (listening)', '312': 'Break between classes', '313': 'School homework, self-study (individual)', '314': 'Team project (in groups)', '321': 'Private tutoring (offline)', '322': 'Online courses', '41': 'Preparing food and washing dishes', '42': 'Laundry and ironing', '43': 'Housing management and cleaning', '44': 'Vehicle management', '45': 'Pet and plant caring', '46': 'Purchasing goods and services (grocery/take-out)', '51': 'Caring for children under 10 who live together', '52': 'Caring for elementary, middle, and high school students over 10 who live together', '53': 'Caring for a spouse', '54': 'Caring for parents and grandparents who live together', '55': 'Caring for other family members who live together', '56': 'Caring for parents and grandparents who do not live together', '57': 'Caring for other family members who do not live together', '81': 'Personal care-related travel', '82': 'Commuting and work-related travel', '83': 'Education-related travel', '84': 'Travel related to housing management', '85': 'Travel related to caring for family and household members', '86': 'Travel related to participation and volunteering', '87': 'Socializing and leisure-related travel', '61': 'Religious activities', '62': 'Political activity', '63': 'Ceremonial activities', '64': 'Volunteer', '711': 'Offline communication', '712': 'Video or voice call', '713': 'Text or email (Online)', '721': 'Reading books, newspapers, and magazines', '722': 'Watching TV or video', '723': 'Listening to audio', '724': 'Internet search or blogging', '725': 'Gaming (mobile, computer, video)', '741': 'Watching a sporting event', '742': 'Watching movie', '743': 'Concerts and plays', '744': 'Amusement Park, zoo', '745': 'Festival, carnival', '746': 'Driving, sightseeing, excursion', '751': 'Walking', '752': 'Running, jogging', '753': 'Climbing, hiking', '754': 'Biking', '755': 'Ball games (soccer, basketball, baseball, tennis, etc)', '756': 'Camping, fishing', '761': 'Group games (board games, card games, puzzles, etc.)', '762': 'Personal hobbies (woodworking, gardening, etc.)', '763': 'Group performances (orchestra, choir, troupe, etc.)', '764': 'Liberal arts and learning (languages, musical instruments, etc.)', '791': 'Nightlife', '792': 'Smoking', '793': 'Do nothing and rest', '91': 'Online shopping', '92': 'Offline shopping'}
PERSON_DIRS = [0,6,10,12,20,25,30]

def main():
    ########## PARSE ARUGMENTS ###########
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default="lstm")
    parser.add_argument('--data_dir', type=str, default="/data/etri_lifelog")
    parser.add_argument('--act_flag', type=str, default="act")
    parser.add_argument('--person_index', type=int, default=1)
    parser.add_argument('--split_ratio', type=float, default=0.67)
    parser.add_argument('--use_timestamp', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=41)
    
    # for MLP, LSTM, and biLSTM
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--sequence_length_lstm', type=int, default=10)
    parser.add_argument('--sequence_length_bilstm', type=int, default=15)

    # for transformers
    parser.add_argument('--d_hid', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--nlayers', type=int, default=3)
    parser.add_argument('--sche', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--sequence_length_transformer', type=int, default=10)
    args = parser.parse_args()

    fix_seed(args.seed)
    ############## LOAD DATA #########s#####    
    person_dir_index = 0
    while person_dir_index < len(PERSON_DIRS) and PERSON_DIRS[person_dir_index] < args.person_index:
        person_dir_index += 1
    data_path="user{0:02d}-{1:02d}/user{2:02d}".format(PERSON_DIRS[person_dir_index-1]+1, PERSON_DIRS[person_dir_index], args.person_index)
    print(data_path)

    data_path = os.path.join(args.data_dir, data_path)
    _, _, _, test_feat_lstm, test_label_act_lstm, test_label_emotion_lstm, num_classes_lstm = load_data_lstm(data_path, split_ratio=args.split_ratio, act_flag=args.act_flag, seq_len=args.sequence_length_lstm)
    _, _, _, test_feat_bilstm, test_label_act_bilstm, test_label_emotion_bilstm, num_classes_bilstm = load_data_lstm(data_path, split_ratio=args.split_ratio, act_flag=args.act_flag, seq_len=args.sequence_length_bilstm)
    _, _, _, test_feat_mlp, test_label_act_mlp, test_label_emotion_mlp, num_classes_mlp = load_data_mlp(data_path, split_ratio=args.split_ratio, act_flag=args.act_flag) # seq_length == 1
    _, _, _, test_feat_transformer, test_label_act_transformer, test_label_emotion_transformer, num_classes = load_data_transformer(data_path, split_ratio=args.split_ratio, act_flag=args.act_flag, seq_len=args.sequence_length_transformer)
        
    assert num_classes_lstm == num_classes_bilstm == num_classes_mlp == num_classes
    
    with torch.cuda.device(0):
        test_feat_lstm = test_feat_lstm.cuda()
        test_label_act_lstm = test_label_act_lstm.cuda()
        test_label_emotion_lstm = test_label_emotion_lstm.cuda()
        test_feat_bilstm = test_feat_bilstm.cuda()
        test_label_act_bilstm = test_label_act_bilstm.cuda()
        test_label_emotion_bilstm = test_label_emotion_bilstm.cuda()
        test_feat_mlp = test_feat_mlp.cuda()
        test_label_act_mlp = test_label_act_mlp.cuda()
        test_label_emotion_mlp = test_label_emotion_mlp.cuda()
        test_feat_transformer = test_feat_transformer.cuda()
        test_label_act_transformer = test_label_act_transformer.cuda()
        test_label_emotion_transformer = test_label_emotion_transformer.cuda()

    ########### INITIALIZE MODEL ###########

    ## load LSTM
    lstm = LSTM(num_classes, test_feat_lstm.shape[-1], args.hidden_size, args.num_layers, bidirectional_flag=False, dropout=args.dropout).to(0)
    exp_name = './wgt/LSTM_USER_%s_c%s_lr1e-05_wd0.01_dr0.4'%(args.person_index, num_classes)
    print (exp_name)
    checkpoint = torch.load(exp_name+'/best.ckpt')
    lstm.load_state_dict(checkpoint['model_state_dict'])

    ## load biLSTM
    bilstm = LSTM(num_classes, test_feat_bilstm.shape[-1], args.hidden_size, args.num_layers, bidirectional_flag=True, dropout=args.dropout).to(0)    
    exp_name = './wgt/LSTM_USER_%s_c%s_lr1e-05_wd0.01_dr0.4_bi'%(args.person_index, num_classes)
    print (exp_name)
    checkpoint = torch.load(exp_name+'/best.ckpt')
    bilstm.load_state_dict(checkpoint['model_state_dict'])
    
    ## load MLP
    mlp = MLP(num_classes, test_feat_mlp.shape[-1], args.hidden_size, args.num_layers, args.dropout).to(0)
    exp_name = './wgt/MLP_USER_%s_c%s_lr1e-05_wd0.01_dr0.4'%(args.person_index, num_classes)
    print (exp_name)
    checkpoint = torch.load(exp_name+'/best.ckpt')
    mlp.load_state_dict(checkpoint['model_state_dict'])
    
    ## load Transformer
    transformer = TransformerBased(test_feat_transformer.shape[-1], args.d_model, args.nhead, args.d_hid, args.nlayers, num_classes, args.dropout).to(0)
#     exp_name = './wgt/USER_%s_c%s_dh16_dm16_he4_l3_lr1e-05_wd0.01_dr0.4_nosche_s10'%(args.person_index, num_classes)
    exp_name = './wgt/USER_%s_c%s_dh16_dm16_he4_l3_lr1e-05_wd0.01_dr0.4_nosche_s10_es41_fin_mep300'%(args.person_index, num_classes)
    print (exp_name) # ./wgt/USER_1_c15_dh16_dm16_he4_l3_lr1e-05_wd0.01_dr0.4_nosche_s10
    checkpoint = torch.load(exp_name+'/best.ckpt')
    transformer.load_state_dict(checkpoint['model_state_dict'])
    criterion_emotion = nn.HuberLoss()
    
    with torch.no_grad():
        lstm.eval()
        bilstm.eval()
        mlp.eval()
        transformer.eval()
        
        ## eval multiple models:
        test_pred_mlp = mlp(test_feat_mlp)
        pred_act_mlp, pred_emotion_mlp = test_pred_mlp
        test_pred_lstm = lstm(test_feat_lstm)
        pred_act_lstm, pred_emotion_lstm = test_pred_lstm
        test_pred_bilstm = bilstm(test_feat_bilstm)
        pred_act_bilstm, pred_emotion_bilstm = test_pred_bilstm
        
        # for transformer (using own dataloader)
        pred_act_transformer_list = []
        gt_act_transformer_list = []
        pred_emotion_transformer_list = []
        src_mask = generate_square_subsequent_mask(args.sequence_length_transformer).to(0)
        
        for i in range(len(test_feat_transformer)):
            test_pred = transformer(test_feat_transformer[i].unsqueeze(dim=1), src_mask)
            pred_act_transformer, pred_emotion_transformer = test_pred
            if i == 0:
                for j in range(pred_act_transformer.shape[0]):
                    pred_act_transformer_list.append(pred_act_transformer[j])
                    gt_act_transformer_list.append(test_label_act_transformer[i][j])
                    pred_emotion_transformer_list.append(pred_emotion_transformer[j])
            else:
                pred_act_transformer_list.append(pred_act_transformer[-1])
                gt_act_transformer_list.append(test_label_act_transformer[i][-1])
                pred_emotion_transformer_list.append(pred_emotion_transformer[-1])

        test_label_act_transformer = torch.Tensor(gt_act_transformer_list)
        pred_act_transformer = torch.stack(pred_act_transformer_list)
        pred_emotion_transformer = torch.stack(pred_emotion_transformer_list)
        
        len_data_bilstm = len(test_label_act_bilstm)
        len_data_mlp = len(test_label_act_mlp)
        len_data_lstm = len(test_label_act_lstm)
        len_data_transformer = len(test_label_act_transformer)

        assert test_label_act_mlp[len_data_mlp-len_data_bilstm:].sum() == test_label_act_lstm[len_data_lstm-len_data_bilstm:].sum() == test_label_act_bilstm.sum() == test_label_act_transformer[len_data_transformer-len_data_bilstm:].sum()
        
        pred_act_mlp = pred_act_mlp[len_data_mlp-len_data_bilstm:]
        pred_act_lstm = pred_act_lstm[len_data_lstm-len_data_bilstm:]
        pred_act_transformer = pred_act_transformer[len_data_transformer-len_data_bilstm:]
       
        # calculate emotion loss:
        loss_emo_mlp = criterion_emotion(pred_emotion_mlp[len_data_mlp-len_data_bilstm:].squeeze(), test_label_emotion_bilstm).cpu()
        loss_emo_lstm = criterion_emotion(pred_emotion_lstm[len_data_lstm-len_data_bilstm:].squeeze(), test_label_emotion_bilstm).cpu()
        loss_emo_bilstm = criterion_emotion(pred_emotion_bilstm.squeeze(), test_label_emotion_bilstm).cpu()
        loss_emo_transformer = criterion_emotion(pred_emotion_transformer[len_data_transformer-len_data_bilstm:].squeeze(), test_label_emotion_bilstm).cpu()
        loss_list = [loss_emo_mlp, loss_emo_lstm, loss_emo_bilstm, loss_emo_transformer]
        np.save(os.path.join('./ensemble', "USER_%s_emo_losses.npy"%(args.person_index)), np.array(loss_list))
         
        pred_act_prob_mlp = nn.Softmax(dim=-1)(pred_act_mlp)
        pred_act_idx_mlp = torch.argmax(pred_act_prob_mlp, axis=-1)
        f = open(os.path.join('./prediction', "USER_%s_pred_act_mlp.csv"%(args.person_index)), "w")
        csv_writer = csv.writer(f)
        csv_writer.writerow(np.array(pred_act_idx_mlp.cpu()))
        f.close()
    
        pred_act_prob_lstm = nn.Softmax(dim=-1)(pred_act_lstm)
        pred_act_idx_lstm = torch.argmax(pred_act_prob_lstm, axis=-1)
        f = open(os.path.join('./prediction', "USER_%s_pred_act_lstm.csv"%(args.person_index)), "w")
        csv_writer = csv.writer(f)
        csv_writer.writerow(np.array(pred_act_idx_lstm.cpu()))
        f.close()
        
        pred_act_prob_bilstm = nn.Softmax(dim=-1)(pred_act_bilstm)
        pred_act_idx_bilstm = torch.argmax(pred_act_prob_bilstm, axis=-1)
        f = open(os.path.join('./prediction', "USER_%s_pred_act_bilstm.csv"%(args.person_index)), "w")
        csv_writer = csv.writer(f)
        csv_writer.writerow(np.array(pred_act_idx_bilstm.cpu()))
        f.close()
        
        pred_act_prob_transformer = nn.Softmax(dim=-1)(pred_act_transformer)
        pred_act_idx_transformer = torch.argmax(pred_act_prob_transformer, axis=-1)
        f = open(os.path.join('./prediction', "USER_%s_pred_act_transformer.csv"%(args.person_index)), "w")
        csv_writer = csv.writer(f)
        csv_writer.writerow(np.array(pred_act_idx_transformer.cpu()))
        f.close()
        
        f = open(os.path.join('./prediction', "USER_%s_gt_act.csv"%(args.person_index)), "w")
        csv_writer = csv.writer(f)
        csv_writer.writerow(np.array(test_label_act_bilstm.cpu()))
        f.close()        

        ## make ensemble model
        # pred_act: [n_instances, n_classes] > [n_instances, n_classes, 3]   
        pred_act_probs = torch.stack((pred_act_prob_mlp, pred_act_prob_bilstm, pred_act_prob_transformer), axis=-1)   

        ## calculate accuracy of ensemble model:
        pred_act_probs_aver = torch.mean(pred_act_probs, axis=-1)        
        k_list = [1, 3, 5]
        act_accu_list = []
        for k in k_list:
            act_accu_list.append(top_k(pred_act_probs_aver.cpu(), test_label_act_bilstm.cpu(), k))
        
        np.save(os.path.join('./ensemble', "USER_%s_best_accu.npy"%(args.person_index)), np.array(act_accu_list))
        print (act_accu_list)
        
        f = open("./ensemble/res_event_det_ensemble_USER%s.txt"%(args.person_index), 'w')
        pred_act_probs = pred_act_probs.detach().cpu().numpy()
        bald_scores = BALD().compute_score(pred_act_probs)
        bald_rank = BALD()(pred_act_probs)
        if np.sum(bald_scores) != 0:
#             print('Scores using bald :', bald_scores) 
            print('Ranks using bald :', bald_rank)

        # All uncertainty scores all zeros.
        var_scores = Variance().compute_score(pred_act_probs)
        var_rank = Variance()(pred_act_probs)
        if np.sum(var_scores) != 0:
#             print('Scores using variance :', var_scores)
            print('Ranks using variance :', var_rank)
    
#         print('Scores using entropy :', Entropy().compute_score(pred_act_probs))
        print('Ranks using entropy :', Entropy()(pred_act_probs))
#         print('Scores using margin :', Margin().compute_score(pred_act_probs))
        print('Ranks using margin :', Margin()(pred_act_probs))
        
        f.write("### RESULTS FROM BALD \n")
        for i in range(10):
            f.write("Top " + str(i+1)+ "/" + str(len(test_label_act_bilstm)) + " uncertain estimation detected by BALD \n")
            sample_idx = bald_rank[i]
            f.write("Sample idx: " + str(sample_idx) + "\n")
            f.write("Scores: " + str(bald_scores[sample_idx]) + "\n")
            f.write("Label: " + str(test_label_act_bilstm[sample_idx]) + "\n")
            f.write("Prediction by ensemble: " + str(torch.argmax(pred_act_probs_aver[sample_idx])) + str(torch.max(pred_act_probs_aver[sample_idx])) + "\n")
            f.write("Probabilities (sanity check**): " + str(pred_act_probs_aver[sample_idx]) + "\n")
            f.write("Prediction by mlp: " + str(torch.argmax(pred_act_prob_mlp[sample_idx])) + str(torch.max(pred_act_prob_mlp[sample_idx])) + "\n")
            f.write("Prediction by lstm: " + str(torch.argmax(pred_act_prob_lstm[sample_idx])) + str(torch.max(pred_act_prob_lstm[sample_idx]))+ "\n")
            f.write("Prediction by bilstm: " + str(torch.argmax(pred_act_prob_bilstm[sample_idx])) + str(torch.max(pred_act_prob_bilstm[sample_idx]))+ "\n")
            f.write("Prediction by transformer: " + str(torch.argmax(pred_act_prob_transformer[sample_idx])) + str(torch.max(pred_act_prob_transformer[sample_idx])) + "\n")
        f.close()

if __name__=='__main__':
    main()
