import pandas as pd
import pickle as pkl
 
with open ('./CA1883.pkl','rb') as file:
    data=pkl.load(file)
print(data)
print(type(data))
# df=pd.DataFrame(data[5])
# print(df)
# df.to_csv('output_file2.csv',index=False,header=False)

