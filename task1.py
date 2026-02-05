import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

train_df=pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")

train_df["TotalBathrooms"]=train_df["FullBath"]+0.5*train_df["HalfBath"]
test_df["TotalBathrooms"]=test_df["FullBath"]+0.5*test_df["HalfBath"]

x_train=train_df[["GrLivArea","BedroomAbvGr","TotalBathrooms"]]
y_train=train_df["SalePrice"]

x_test=test_df[["GrLivArea","BedroomAbvGr","TotalBathrooms"]]

x_train=x_train.fillna(x_train.median())
x_test=x_test.fillna(x_test.median())

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

model=LinearRegression()
model.fit(x_train_scaled,y_train)

test_predictions=model.predict(x_test_scaled)

output=pd.DataFrame({"Id":test_df["Id"],"SalePrice":test_predictions})

output.to_csv("submissions.csv",index=False)
print("File saved as submissions.csv")

train_predictions=model.predict(x_train_scaled)

mse=mean_squared_error(y_train,train_predictions)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_train,train_predictions)
r2=r2_score(y_train,train_predictions)

print("MSE: ",mse)
print("RMSE :", rmse)
print("MAE :", mae)
print("R2 Score :", r2)

plt.figure(figsize=(8,6))
plt.scatter(y_train,train_predictions,alpha=0.4,s=25)
plt.xlabel("Actual Sale Prices",fontsize=12)
plt.ylabel("Predicted Sale Prices",fontsize=12)
plt.title("Actual vs Predicted Prices",fontsize=14)
plt.plot([y_train.min(),y_train.max()],[y_train.min(),y_train.max()],linewidth=2)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()