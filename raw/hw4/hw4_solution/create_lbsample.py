import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from extractFeat import extract_date, extract_company, preprocess_data


def reorder_columns(df, target='3_DAY_RETURN'):
    df = df[[col for col in df.columns if col != target] + [target]]
    return df


sample1 = pd.read_csv("sample1.csv")
rng = np.random.default_rng(seed=334)

# sample half of it
sidx = rng.choice(range(sample1.shape[0]),
					    size=50000)

ssdf = sample1.iloc[sidx, :]
# split into train and test
lbtrain, lbtest = train_test_split(ssdf, random_state=334,
								   test_size=0.2)

lbtrain.to_csv("lb_train.csv", index=False)
lbtest.to_csv("lb_test.csv", index=False)

# add the dataset
proc_train = extract_date(lbtrain)
proc_test = extract_date(lbtest)
# extract company features for both/test
proc_train = extract_company(proc_train)
proc_test = extract_company(proc_test)
# run pre-process
proc_train, proc_test = preprocess_data(proc_train, proc_test)

# drop the text data
proc_train.drop(columns=["TWTOKEN", "TWEET"], inplace=True)
proc_test.drop(columns=["TWTOKEN", "TWEET"], inplace=True)
# do some book-keeping here to make sure columns are the same
ptrain, ptest = proc_train.align(proc_test,
                                 join='outer',
                                 axis=1, fill_value=0)
ptrain = reorder_columns(ptrain)
ptest = reorder_columns(ptest)

lr = LinearRegression()
lr.fit(ptrain.iloc[:, :-1], ptrain.iloc[:, -1])
yhat = lr.predict(ptest.iloc[:, :-1])
ytrue = ptest.iloc[:, -1]

mean_squared_error(ytrue, yhat)
