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
    desc = 'Process dataset for literature review query'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('dataset',
                       metavar='dataset',
                       type=str,
                       help='Path to the dataset to process')
    parser.add_argument('titles',
                       metavar='titles',
                       type=str,
                       help='Name of column containing titles')
    parser.add_argument('abstracts',
                       metavar='abstracts',
                       type=str,
                       help='Name of column containing abstracts')
    parser.add_argument('authors',
                       metavar='authors',
                       type=str,
                       help='Name of column containing authors')
    parser.add_argument('questions',
                       metavar='questions',
                       type=str,
                       help='Name of column containing research questions')
    parser.add_argument('results',
                       metavar='results',
                       type=str,
                       help='Path to the write the resulting dataset')

    args = parser.parse_args()
    data_path = args.dataset
    title_col = args.titles
    abstract_col = args.abstracts
    question_col = args.questions
    author_col = args.authors
    results = args.results

    if not os.path.isfile(data_path):
        print(" ERROR")
        print(" The input file '%s' does not exist" % data_path)
        sys.exit()

    df = process_data(data_path, title_col, abstract_col, author_col, question_col)

    df.to_csv(results, header=True, index=False)


#################################################################################
def process_data(data_path, title_col, abstract_col, author_col, question_col):
    """
    Main control function
    Load the data and then add the feature sets.
    Gradually building up a larger feature enriched dataframe.
    """
    df = pd.read_csv(data_path, encoding="ISO-8859-1")
    df2 = add_text_summary_features(df, title_col, abstract_col)
    df3 = add_author_features(df2, author_col)
    df4 = add_question_match_features(df3, "P", "TEXT")
    df5 = add_question_match_features(df4, "I", "TEXT")
    df6 = add_question_match_features(df5, "C", "TEXT")
    df7 = add_question_match_features(df6, "O", "TEXT")
    return df7


#################################################################################
def intersection(lst, refset):
    lst3 = [value for value in lst if value in refset]
    return len(set(lst3))

def prop_intersection(lst, refset, rounder=3):
    inter = intersection(lst, refset)
    return round(inter/len(refset), rounder)

#################################################################################
def add_question_match_features(df, q_col, t_col, sfx=""):
    """
    Return a copy of a dataframe with features describing matching
    between the query keywords and the articles in the dataframe
    """
    df_new = df.copy()
    def q_feats(x):
      try:
        crit = x[q_col]
        question = crit.lower().translate(str.maketrans('', '', string.punctuation))
        kwds = set(question.split(" ")) - set(stop_word_list)
        raw_text = x[t_col]
        txt = raw_text.lower().translate(str.maketrans('', '', string.punctuation))
        wds = txt.split(" ")
        inter = prop_intersection(wds, kwds)
        jd = jellyfish.jaro_distance(raw_text,crit)
        ld = jellyfish.levenshtein_distance(raw_text,crit)
        ji = textdistance.jaccard(raw_text,crit)
        sd = textdistance.sorensen(raw_text, crit)
        return inter, jd, ld, ji, sd
      except:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    col_pfix = q_col + "_" + t_col + sfx
    df_new[[col_pfix+'_match_prop', col_pfix+'_jd', col_pfix+'_ld', col_pfix+'_ji', col_pfix+'_sd']] = df_new.apply(q_feats, axis=1, result_type="expand")
    return df_new

#################################################################################
def add_author_features(df, author_col, sepa=";"):
    """
    Return a copy of a dataframe with summary features added for
    the named columns for the author of the paper.
    """
    df_new = df.copy()
    def author_features(x, col):
        try:
           authors = str(x[col])
           if authors=="Anonymous":
               return 0
           authors = authors.replace(',', ';') 
           auth_array = authors.split(sepa)
           return len(auth_array)
        except:
           return 0
    df_new['author_count'] = df_new.apply(author_features, col=author_col, axis=1)
    return df_new
 

#################################################################################
def add_text_summary_features(df, title_col, abstract_col):
    """
    Return a copy of a dataframe with summary features added for
    the named columns for the title and abstract of the paper.
    """
    df_new = df.copy()
    def len_or_null(x):
        if x == x:
            return len(x)
        else:
            return np.nan

    df_new['title_length'] = df_new[title_col].apply(len_or_null)
    df_new['abstract_length'] = df_new[abstract_col].apply(len_or_null)

    def get_str(x):
        if x != x:
            return ""
        else:
            return str(x)

    def text_gen(x):
           return get_str(x[title_col]) + " " + get_str(x[abstract_col])

    df_new['TEXT'] = df_new.apply(text_gen, axis=1)
    df_new['TEXT_length'] = df_new['TEXT'].apply(len_or_null)

    def text_features(x, col):
        if x[col] != x[col]:
           return np.nan, np.nan, np.nan, np.nan
        elif x[col] == "":
           return 0, 0, 0, 0
        else:
           content = str(x[col])
           word_array = content.lower().split()
           non_stop_words = list(set(word_array) - set(stop_word_list))
           word_count = len(word_array)
           word_lengths = list(map(len, word_array))
           if word_count > 0:
             max_word_len = max(word_lengths)
             avg_word_len = sum(word_lengths)/word_count
             content_wd = len(non_stop_words)/len(word_array)
           else:
             max_word_len = 0
             avg_word_len = 0
             content_wd = 0
           return word_count, max_word_len, avg_word_len, content_wd
    df_new[['title_wc', 'title_max_wl', 'title_avg_wl', 'title_cwd']] = df_new.apply(text_features, col=title_col, axis=1, result_type="expand")    
    df_new[['abstract_wc', 'abstract_max_wl', 'abstract_avg_wl', 'abstract_cwd']] = df_new.apply(text_features, col=abstract_col, axis=1, result_type="expand")
    df_new[['TEXT_wc', 'TEXT_max_wl', 'TEXT_avg_wl', 'TEXT_cwd']] = df_new.apply(text_features, col='TEXT', axis=1, result_type="expand")
    return df_new
 


#################################################################################
if __name__ == "__main__": 
    main()


