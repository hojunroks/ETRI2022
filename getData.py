import os
import pandas as pd
import pdb


data_path = "/root/storage/etri_lifelog/user01-06/user01"
dir_list = os.listdir(data_path)

user_all_data = []

for item in dir_list:
    csv_file = data_path + "/" + item + "/" + item + "_label.csv"
    # csv_file = "/root/storage/etri_lifelog/user01-06/user01/1598759880/1598759880_label.csv"

    f = open(csv_file,'r')
    rdr = pd.read_csv(f)
    user_all_data.append(rdr)

    # ts,action,actionOption,actionSub,actionSubOption,condition,conditionSub1Option,conditionSub2Option,place,emotionPositive,emotionTension,activity

    df = pd.DataFrame(rdr, columns=rdr.columns)

    corr = df.corr(method="pearson")
    print(corr)

print("\nUSER 01 ALL DATA CORRELATIONS\n")
df_all = pd.concat(user_all_data, axis=0)
corr = df_all.corr(method="pearson")
print(corr)