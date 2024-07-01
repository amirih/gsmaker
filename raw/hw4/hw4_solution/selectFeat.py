import argparse
import pandas as pd
# solution
import numpy as np


def cal_corr(df):
    """
    Compute the Pearson correlation matrix
    
    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    corrDF : pandas.DataFrame
        The correlation between the different columns
    """
    corrDF = df.corr()
    return corrDF


def select_features(trainDF, testDF):
    """
    Preprocess the features
    
    Parameters
    ----------
    trainDF : pandas.DataFrame
        the training dataframe
    testDF : pandas.DataFrame
        the test dataframe

    Returns
    -------
    trainDF : pandas.DataFrame
        return the feature-selected trainDF dataframe
    testDF : pandas.DataFrame
        return the feature-selected testDT dataframe
    """
    abs_corr = cal_corr(trainDF).abs()
    # first drop the uncorrelated features to the target
    drop_columns = abs_corr.iloc[:, -1] >= 0.01
    # drop the relevant columns in train and test
    trainDF = trainDF.loc[:, drop_columns]
    testDF = testDF.loc[:, drop_columns]
    # re-compute and convert to numpy
    corr_df = cal_corr(trainDF).abs()
    np.fill_diagonal(corr_df.values, 0)
    corr_mat = np.triu(corr_df)
    # set the last one to be 0's to avoid removing things
    # highly correlated with the target
    corr_mat[:, -1] = 0
    select_indices = zip(*np.where(corr_mat > 0.60))
    drop_columns = set([])
    for i, j in select_indices:
        # determine which one has higher 
        if corr_mat[i, -1] > corr_mat[j, -1]:
            # drop j
            drop_columns.add(j)
        else:
            drop_columns.add(i)
    trainDF.drop(columns=trainDF.columns[list(drop_columns)],
                 inplace=True)
    testDF.drop(columns=testDF.columns[list(drop_columns)],
                inplace=True)
    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("inTrain",
                        help="filename of the training data")
    parser.add_argument("inTest",
                        help="filename of the test data")
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")

    args = parser.parse_args()
    # load the train and test data
    train_df = pd.read_csv(args.inTrain)
    test_df = pd.read_csv(args.inTest)

    print("Original Training Shape:", train_df.shape)
    # calculate the training correlation
    train_df, test_df = select_features(train_df,
                                        test_df)
    print("Transformed Training Shape:", train_df.shape)
    # save it to csv
    train_df.to_csv(args.outTrain, index=False)
    test_df.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()



