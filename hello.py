import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt

import os
from os import listdir

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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
# print(df.info())

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


# Độ tương quan giữa giá xe và tổng số dặm đã đi
"""
plt.style.use("seaborn-v0_8")
plt.title('Price vs Mileage')
plt.scatter(data=df, x='mpg', y='price')
plt.show() # Chưa thấy tương quan nhiều -> Thử logarit biến mileage
"""

df['ln_mileage'] = df['mileage'].apply(lambda x: math.log(x))
# plt.title('Price vs ln_mileage')
# plt.scatter(data=df, x='ln_mileage', y='price')
# plt.show()

# print(df[['ln_mileage', 'price']].corr()) # Tương quan nghịch -> Đúng với nhận định ban đầu

# Độ tương quan giữa giá xe và số dặm đi được trên 1 gallon (mpg)
"""
plt.style.use('seaborn-v0_8')
plt.title('Price vs Mpg')
plt.scatter(data=df, x='mpg', y='price')
plt.show() # Tương tự
"""

df['ln_mpg'] = df['mpg'].apply(lambda x: math.log(x))
# plt.title('Price vs ln_mpg')
# plt.scatter(data=df, x='ln_mpg', y='price')
# plt.show()

# print(df[['ln_mpg', 'price']].corr())

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
# Hiển thị missing value
"""
columns = df.columns
for i,column in enumerate(columns):
  missing = df[column].isnull().sum()
  percent = missing*100/(df.shape[0])
  if missing  > 0:
    print('Column', column, 'contains', missing, 'missing value')
    print('take', format(percent, ".2f"), "%")
    print("--------------------")
"""

# Thay giá trị bị null ở cột tax và ln_mpg bằng giá trị trung bình của toàn dữ liệu
avg = df[df['ln_mpg'].notnull()]['ln_mpg'].mean() # Tính giá trị trung bình
df['ln_mpg'] = df['ln_mpg'].fillna(avg)
# print(df['ln_mpg'])
df['tax'] = df['tax'].fillna(df[df['tax'].notnull()]['tax'].mean())
# print(df['tax'])

# Data Transformation - Chuyển đổi dữ liệu
# print(df.head()) # Dữ liệu chưa chuyển đổi
def fill_class(x):
    if x in ('merc', 'audi', 'bmw'):
        return "Luxury"
    if x in ('vw', 'skoda', 'focus'):
        return "Mid-range"
    else:
        return "Affordable"

df['Class'] = df['automaker'].apply(lambda x: fill_class(x))
# print(df['Class'].value_counts())

# print(df['year'].describe())
df['year'] = pd.cut(df['year'],
                    bins=[0, 2015.9, 2016.9, 2018.9, 2061],
                    labels=['hơn 5 năm', '4-5 năm', '2-4 năm', 'dưới 2 năm'])
# print(df['year'].value_counts())

# print(df['engineSize'].describe())
df['engineSize'] = pd.cut(df['engineSize'],
                          bins = [-1, df['engineSize'].quantile(0.25)-0.01, df['engineSize'].quantile(0.5)-0.01, df['engineSize'].quantile(0.75)-0.01, df['engineSize'].max()+0.01],
                          labels=['Small','Medium','Large','Very Large'])
# print(df['engineSize'].value_counts())                    

# Xử lí Outlier
cols = ['ln_mileage','tax','ln_mpg','price']
for col in cols:
    # Tính IQR của biến
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Xác định ngưỡng trên và ngưỡng dưới
    upper_threshold = Q3 + (1.5 * IQR)
    lower_threshold = Q1 - (1.5 * IQR)

    # Thay thế các giá trị outlier
    df[col] = df[col].apply(lambda x: upper_threshold if x >= upper_threshold else (lower_threshold if x <= lower_threshold else x))
# print(df.head())

# Chia tập dữ liệu X, y
cat_features = ['year','transmission','fuelType','engineSize','Class']
num_features = ['ln_mileage','tax','ln_mpg']
features = cat_features + num_features

X = df[features].reset_index(drop=True)
y = df['price'].reset_index(drop=True)
# print(X.head())

# chuẩn hóa và xử lý dữ liệu
# Đường ống 
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
cat_transformer = Pipeline([('encoder',OneHotEncoder())])

# Khai báo quá trình xử lý
preprocessor = ColumnTransformer([
    ('num',num_transformer,num_features),
    ('cat',cat_transformer,cat_features),
])

X_transformed = preprocessor.fit_transform(X) # Fit và transform vào bộ dữ liệu

# print(X_transformed)

X_train, X_test, y_train, y_test = train_test_split(X_transformed,y,train_size=0.8) # Chia training set và test set (?)
# print(len(X_train), len(X_test))

# Chạy mô hình
linear_regression = linear_model.LinearRegression() # Khai báo mô hình
linear_regression.fit(X_train, y_train) # Huấn luyện mô hình

y_pred = linear_regression.predict(X_test)
r2 = r2_score(y_test, y_pred)
# print('R2 Score:',r2)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
# print('MSE:',mse)
# print('RMSE:',rmse)

# Hiện đồ thị
"""
y_pred = linear_regression.predict(X_test)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
"""

# Dự báo
# Mô hình HQTT
# Hiển thị các tham số ước lượng
intercept = linear_regression.intercept_
coef = linear_regression.coef_
# print("Intercept:", intercept)
# print("Coefficients:", coef)

y_pred = linear_regression.predict(X_test)
# print("y_pred:", y_pred)
# print("y_test", y_test)

