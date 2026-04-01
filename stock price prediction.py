import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

## preprocessing the data
def prepare_data(df, target_col, target_out,test_size):
    label=df[target_col].shift(-target_out)
    x=np.array(df[target_col]).reshape(-1,1)
    X=StandardScaler().fit_transform(x)
    X_lately=X[-target_out:]
    X=X[:-target_out]
    label.dropna(inplace=True)
    y=np.array(label)
    x1=int(len(X)*(1-test_size))
    x_train=X[:x1]
    x_test=X[x1:]
    y_train=y[:x1]
    y_test=y[x1:]
    result= [x_train,x_test,y_train,y_test,X_lately]
    return result

## getting data
df=pd.read_csv("prices.csv")
n_column=df[df.symbol=="GOOG"]

# fetching prepared data
x_train,x_test,y_train,y_test,x_lately=prepare_data(n_column,"close",5,0.2)

#evaluating model
model=LinearRegression()
model.fit(x_train,y_train)

#prediction

prediction=model.predict(x_lately)
print(prediction)


print(n_column.tail(5)) ##for matching the accuracy of out prediction



    
