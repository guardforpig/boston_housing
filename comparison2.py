import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
np.set_printoptions(threshold=np.inf)

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
x= np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y= raw_df.values[1::2, 2]
scaler= MinMaxScaler()
x_scaled=scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled,y,random_state=80,test_size=0.5)
#载入线性回归模型
lr=LinearRegression()
#训练
lr.fit(x_train,y_train)
#测试
y_test_predict=lr.predict(x_test)
#画图
plt.plot(y_test_predict,"r-")
plt.plot(y_test,"b-")
plt.legend()
plt.show()

