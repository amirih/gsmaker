import pandas as pd


def reorder_columns(df, target='3_DAY_RETURN'):
    """
    Re-order the columns so the target is the last one
    """
    df = df[[col for col in df.columns if col != target] + [target]]
    return df


def test_preprocess_data(df):
    # extract date
    df['DATE'] = pd.to_datetime(df['DATE'],
                                dayfirst=True)
    # convert month
    df['MONTH'] = df['DATE'].dt.month
    # extract day of the week
    df['DOW'] = df['DATE'].dt.dayofweek
    # extract year
    df['YEAR'] = df['DATE'].dt.year
    df = df.drop(columns=['DATE'])
    # one hot encode
    df = pd.get_dummies(df, columns=['STOCK'])
    df = df.drop(columns=["TWTOKEN", "TWEET"])
    return df



def align_preprocess(traindf, testdf):
    tmp1 = test_preprocess_data(traindf)
    tmp2 = test_preprocess_data(testdf)
    ptrain, ptest = tmp1.align(tmp2,
                               join='outer',
                               axis=1, fill_value=0)
    return reorder_columns(ptrain), reorder_columns(ptest)


def simple_process(df):
    # drop stock and drop date
    df = df.drop(columns=["DATE", "STOCK", "TWTOKEN", "TWEET"])
    return reorder_columns(df)
