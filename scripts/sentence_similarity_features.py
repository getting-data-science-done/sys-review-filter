import os
import sys
import string
import argparse
import pandas as pd
from io import StringIO
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
 
models = {
    "biobert":"pritamdeka/BioBert-PubMed200kRCT",
    "bluebert":"bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    "blubertxl":"bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"
}


####################################################################################
def main():
    desc = 'Process dataset to add senstence similarity column'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('dataset',
                       metavar='dataset',
                       type=str,
                       help='Path to the dataset to process')
    parser.add_argument('model',
                       metavar='model',
                       type=str,
                       help='Name of the BERT model to use: [biobert, bluebert, bluebertxl]')
    parser.add_argument('text',
                       metavar='text',
                       type=str,
                       help='Name of column containing text block')
    parser.add_argument('question',
                       metavar='question',
                       type=str,
                       help='Name of column containing research questions')
    parser.add_argument('results',
                       metavar='results',
                       type=str,
                       help='Path to the write the resulting dataset')

    args = parser.parse_args()
    data_path = args.dataset
    model_name = args.model
    text_col = args.text
    question_col = args.question
    results = args.results

    if not os.path.isfile(data_path):
        print(" ERROR")
        print(" The input file '%s' does not exist" % data_path)
        sys.exit()

    if model_name in models:
        model = SentenceTransformer(models[model_name])
    else:
        print(" ERROR")
        print(" The model '%s' is not in the list. Use one of [biobert, bluebert, bluebertxl]" % model_name)
        sys.exit()

    df = process_data(data_path, model, text_col, question_col)

    df.to_csv(results, header=True, index=False)

    """
    # THIS VERSION JUST PRINTED TO STDOUT
    output = StringIO()
    df.to_csv(output, index=False, header=True)
    output.seek(0)
    print(output.read())
    """


#################################################################################
def process_data(data_path, model, text_col, question_col):
    """
    Main control function
    Load the data and then add the similarity feature.
    """
    df = pd.read_csv(data_path, encoding="ISO-8859-1")
    df2 = add_similarity_feature(df, model, text_col, question_col)
    return df2


#################################################################################
def add_similarity_feature(df, model, col1, col2):
    """
    Return a copy of a dataframe with similarity feature added for the named columns.
    """
    df_new = df.copy()

    def sentence_similarity(x, col1, col2):
        txt1 = x[col1]
        txt2 = x[col2]
        if txt1 != txt1:
            txt1 = ""
        if txt2 != txt2:
            txt2 = ""

        embedding_1 = model.encode(txt1, convert_to_tensor=True)
        embedding_2 = model.encode(txt2, convert_to_tensor=True)
        return util.pytorch_cos_sim(embedding_1, embedding_2).item()

    col_name = "similarity_" + col1 + "__" + col2

    df_new[col_name] = df_new.apply(lambda x: sentence_similarity(x, col1, col2), axis=1)    
    return df_new
 

#################################################################################
if __name__ == "__main__": 
    main()

