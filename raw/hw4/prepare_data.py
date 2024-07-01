import argparse
import pandas as pd
#twitter preprocessor
import preprocessor as xp


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("infile",
                        help="filename of the input data")
    parser.add_argument("outfile",
                        help="filename of the preprocessed data")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.infile)
    # drop a few columns
    raw_df.drop(columns=["Unnamed: 0", 
                         "1_DAY_RETURN", "2_DAY_RETURN",
                         "7_DAY_RETURN"],
                inplace=True)
    if "MENTION" in raw_df.columns:
        raw_df.drop(columns=["MENTION"], inplace=True)
    raw_df = raw_df.dropna() # drop rows with NAN
    raw_df['TWTOKEN'] = raw_df['TWEET'].apply(lambda x: xp.tokenize(x))
    raw_df.to_csv(args.outfile, index=False)


if __name__ == "__main__":
    main()

