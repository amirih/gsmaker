import argparse


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("infile",
                        help="filename of original data")
    parser.add_argument("outfile",
                        help="filename of the reformatted data")
    args = parser.parse_args()
    outfile = open(args.outfile, "w")

    with open(args.infile) as file:
        old_tweet = ""
        header = True
        for tweet in file:
            # deal with the header once
            if header:
                outfile.write(tweet)
                outfile.write("\n")
                header = False
                continue
            # remove strings
            st_tweet = tweet.strip()
            if not st_tweet:
                continue
            if st_tweet[0].isdigit():           
                if old_tweet:
                    outfile.write(old_tweet)
                    outfile.write("\n")
                old_tweet = st_tweet
            else:
                # must be a part of the last one
                old_tweet = old_tweet + st_tweet
        outfile.write(old_tweet)
        outfile.write("\n")


if __name__ == "__main__":
    main()

