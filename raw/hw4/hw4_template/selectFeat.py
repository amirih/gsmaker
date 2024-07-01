import argparse
import pandas as pd


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
    return None


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



