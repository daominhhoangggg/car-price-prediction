import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
import os
from os import listdir

# Ẩn cảnh báo
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Đọc dữ liệu
dirs = os.listdir('data')

basic = 'data/'
df = pd.DataFrame(data= None)
for file in dirs:
    url = basic + file
    new_df = pd.read_csv(url)
    new_df['automaker'] = file.split(".")[0]
    if file == 'hyundi.csv':
        new_df.rename(columns={'tax(£)':'tax'}, inplace=True)
    df = pd.concat([df,new_df])

# Cột object -> categorical
catergorical_cols = ['model','transmission','fuelType','automaker']
for catergorical_col in catergorical_cols:
    df[catergorical_col] = df[catergorical_col].astype('category')

# Cột numerical có ít giá trị riêng biệt sẽ được nhóm cụm -> categorical
numerical_cols = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
distincts = []
percents = []
for numerical_col in numerical_cols:
    distinct = df[numerical_col].nunique()
    percent = distinct*100/df.shape[0]
    distincts.append(distinct)
    percents.append(format(percent,'.2f') + '%')
data = {'Count Distinct': distincts, 'Take Percent':percents}
# print(pd.DataFrame (data=data, index= numerical_cols))


"""
# Độ tương quan giữa giá xe và tổng số dặm đã đi
plt.style.use("seaborn-v0_8")
plt.title('Price vs Mileage')
plt.scatter(data=df, x='mpg', y='price')
plt.show() # Chưa thấy tương quan nhiều -> Thử logarit biến mileage

df['ln_mileage'] = df['mileage'].apply(lambda x: math.log(x))
plt.title('Price vs ln_mileage')
plt.scatter(data=df, x='ln_mileage', y='price')
plt.show()

print(df[['ln_mileage', 'price']].corr()) # Tương quan nghịch -> Đúng với nhận định ban đầu

# Độ tương quan giữa giá xe và số dặm đi được trên 1 gallon (mpg)
plt.style.use('seaborn-v0_8')
plt.title('Price vs Mpg')
plt.scatter(data=df, x='mpg', y='price')
plt.show() # Tương tự

df['ln_mpg'] = df['mpg'].apply(lambda x: math.log(x))
plt.title('Price vs ln_mpg')
plt.scatter(data=df, x='ln_mpg', y='price')

print(df[['ln_mpg', 'price']].corr())
"""


"""
# Đối với biến category
print(df.describe(include=['category']))
print('-------------------------------------------')
print(df.groupby('model')['price'].mean().describe()) # Phân bổ của giá theo hãng xe

# Giá xe trung bình theo transmission, fuelType, automaker(hãng)
cols = ['transmission', 'fuelType', 'automaker']
# data = df.groupby[cols[i]]
n_cols = 2
n_rows = 2
fix, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*3.5, n_rows*3.5))
sns.set_theme(style="whitegrid")
sns.set_color_codes("pastel")
for r in range(n_rows):
    for c in range(n_cols):
        i = r*n_cols + c #Lấy vị trí của cols muốn vẽ
        if i < len(cols):
            ax_i = ax[r,c] #Lấy axis muốn vẽ
            data = df.groupby(cols[i])['price'].mean().sort_values(ascending = False).reset_index()
            palette = sns.color_palette("deep", len(data)) #Tô màu cho các cột
            sns.barplot(data = data ,y=cols[i],x = 'price',ax= ax_i, palette=palette)
            ax.flat[-1].set_visible(False) #Xóa axis cuối
            plt.tight_layout() #Để các đồ thị không đè lên nhau
plt.show()
# Ta thấy xe nhiều hãng có giá trị trung bình tương tự nhau 
# => gom cột Model và cột Automaker thành cột mới và chia theo 'Hạng xe'
"""

# Xử lý mising value
columns = df.columns
for i,column in enumerate(columns):
  missing = df[column].isnull().sum()
  percent = missing*100/(df.shape[0])
  if missing  > 0:
    print('Column', column, 'contains', missing, 'missing value')
    print('take', format(percent, ".2f"), "%")
    print("--------------------")