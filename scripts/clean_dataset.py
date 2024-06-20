import os
import sys
import string
import argparse
import jellyfish
import textdistance
import pandas as pd
from io import StringIO
import numpy as np
from pathlib import Path
from nltk.corpus import stopwords

stop_word_list = stopwords.words('english')

####################################################################################
def main():
    desc = 'Clean out obvious problematic records'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('dataset',
                       metavar='dataset',
                       type=str,
                       help='Path to the dataset to process')

    args = parser.parse_args()
    data_path = args.dataset

    if not os.path.isfile(data_path):
        print(" ERROR")
        print(" The input file '%s' does not exist" % data_path)
        sys.exit()

    df = clean_data(data_path)

    df.to_csv(data_path, header=True, index=False)


#################################################################################
def clean_data(data_path):
    df = pd.read_csv(data_path, encoding="ISO-8859-1")
    df2 = df[ ~np.isnan(df['Decision'])]
    return df2


#################################################################################
if __name__ == "__main__": 
    main()


