import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
x= np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y= raw_df.values[1::2, 2]

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=80,test_size=0.5)
#载入线性回归模型
lr=LinearRegression()
#训练
lr.fit(x_train,y_train)
#测试
y_test_predict=lr.predict(x_test)
y_train_predict=lr.predict(x_train)
#MSE指标
error_test=mean_squared_error(y_test,y_test_predict)
error_train=mean_squared_error(y_train,y_train_predict)
#画图
plt.plot(y_test_predict,"r-",label="predict_value")
plt.plot(y_test,"b-",label="True_value")
plt.legend()
plt.show()

