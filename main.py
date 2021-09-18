# import module

import pandas as pd
import numpy as np
from keras.layers import LeakyReLU
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.optimizers import Adamax
import xgboost as xgb
# from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# read data

df = pd.read_csv("./data/train20210817v2.csv")
df= df.drop(["SeqNo"],axis=1)

# data proccess MinMaxScale

minmax = preprocessing.MinMaxScaler()
data_minmax = minmax.fit_transform(df)
DF = pd.DataFrame(data_minmax)

X = np.array(DF.drop([13],axis=1))
Y = np.array(DF[13])

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1,random_state=85)

# build CNN model

model = Sequential()
model.add(Conv2D(16,(1,5),padding="same",input_shape=(1,13,1),activation=LeakyReLU(alpha=0.1)))
model.add(Conv2D(32,(1,4),padding="same",activation=LeakyReLU(alpha=0.1)))
model.add(MaxPooling2D(pool_size=(1,2)))
model.add(Conv2D(64,(1,3),padding="same",activation=LeakyReLU(alpha=0.1)))
model.add(MaxPooling2D(pool_size=(1,2)))
model.add(Conv2D(128,(1,3),padding="same",activation=LeakyReLU(alpha=0.1)))
model.add(MaxPooling2D(pool_size=(1,3)))
model.add(Flatten())

model.add(Dense(12,activation=LeakyReLU(alpha=0.1)))
model.add(Dense(1,activation=LeakyReLU(alpha=0.1)))

model.compile(loss="mse",optimizer=Adamax(lr=0.0000066))
model.load_weights("./cnn_weight/model.h5")

# load_model = load_model("E://logs/iris_model.h5")


y_CNN = model.predict(x_test.reshape(len(x_test),1,13,1))

# build XGB model

XGB = xgb.XGBRFRegressor()
XGB.n_estimators=74
XGB.max_depth=15
XGB.subsample=0.9
XGB.random_state=87
XGB.fit(x_train,y_train)
y_RF_series = XGB.predict(x_test)

final_data = pd.read_csv("./data/2021test0831.csv")

final_data = final_data.drop(["SeqNo"],axis=1)

final_minmax = preprocessing.MinMaxScaler()
final_data_minmax = final_minmax.fit_transform(final_data)
final_DF = pd.DataFrame(final_data_minmax)

final_array = np.array(final_DF)

# 0.2CNN + 0.8XGB_RF

final_Y = 0.8*XGB.predict(final_array) + 0.2*model.predict(final_array.reshape(7222,1,13,1)).reshape(7222,)

df_result = pd.read_csv("./data/2110999_TestResult.csv")

result_Y = final_Y*86.532 - 2.438

result_Y_df = pd.DataFrame(result_Y)
DF_RESULT = pd.concat([df_result,result_Y_df],axis=1)

DF_RESULT = DF_RESULT.drop(["預測值"],axis=1)

DF_RESULT.rename(columns={0:"預測值"}).to_csv("./result/110999_TestResult.csv",index=False,encoding="UTF-8")










