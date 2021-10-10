import os
import pandas as pd
import matplotlib.pyplot as plt

rel_path = '/Users/camellia/PycharmProjects/dental_disease/yolobcc/result_plot'
exp_ind = ['102', '112', '178']

# for i in exp_ind:
#     os.path.join(rel_path, f'exp{i}')

df_exp102 = pd.read_csv(os.path.join(rel_path, 'exp102/results.csv'))
#print(df_exp102.iloc[0])
print(type(df_exp102))
print(type(df_exp102.iloc[:, 0]))
print(df_exp102.columns)
#print(df_exp102.iloc[:, 0])

#print(df_exp102["epoch"])
# df_exp102.get('epoch')

#plt.plot(df_exp102.iloc[:, 0], df_exp102.iloc[:, 1])

# df_exp102.plot(x='epoch', y='train/box_loss')
# df_102 = pd.DataFrame(df_exp102)

# for col in df_exp102.columns:
#     print(col)

# epoch
# train/box_loss
# train/obj_loss
# train/cls_loss
# metrics/precision
# metrics/recall
# metrics/mAP_0.5
# metrics/mAP_0.5:0.95
# val/box_loss
# val/obj_loss
# val/cls_loss
# x/lr0
# x/lr1
# x/lr2

# df_102.plot(x = 'epoch', y = 'train/box_loss')
# x = df_102['epoch']
# y = df_102['tran/box_loss']
# plt.plot(x, y)
plt.show()
