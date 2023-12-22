import numpy as np
import matplotlib.pyplot as plt
import os

import pickle
import csv
from datetime import datetime
import time
from io import StringIO
from tqdm import tqdm
import pandas as pd

file_cnt=0;
dp="./CA_data/CA1883"
data_dict={}
csv_cnt=0
# timestamp bnum height speed angle longitude latitude
bnum=10001
for fn in os.listdir(dp):
    dataset_path=os.path.join(dp,fn)
    l_l_msg=[]
    ct=0
    with open (dataset_path,"r") as f:
        
        csvReader=csv.reader(f)
        for row in csvReader:
            ct=ct+1
            if ct==1:
                continue
            
            l_l_msg.append([int(row[0]),
                            (bnum),float(row[4]),
                            float(row[5]),float(row[6]),
                            float(row[7]),float(row[8])])
            # print(row)
    print(l_l_msg)
    m_msg=np.array(l_l_msg)
    print(m_msg)
    print("==============================")
    data_dict[csv_cnt]=m_msg
    print(data_dict[csv_cnt])
    print("--------------------------------")
    csv_cnt=csv_cnt+1
    bnum=bnum+1
print(dp)
dp="./CA1883.pkl"
print(dp)
print(type(data_dict))
with open(dp,'wb') as f:
    pickle.dump(data_dict,f)

