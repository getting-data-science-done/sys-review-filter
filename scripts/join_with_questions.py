import string
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


datasets = {
   "PID":"PID_Database.csv",
   "PBRT":"PBRT_Database.csv",
   "CIDP":"CIDP_database.csv",
   "ESG":"ESG_database.csv",
   "EMR":"EMR_database.csv",
   "IORT":"IORT_Database.csv",
   "CARGEL":"Cargel_Database.csv",
   "LMTA":"LMTA_Database.csv",
   "LANB":"LANB_Database.csv",
   "VERT":"VERT_Database.csv"
}

cols = ["Project", "Age", "Authors", "Title", "Abstract", "Question", "Text", "Decision"]

###################################################################

def coerce_year(year):
   if str(year).isnumeric():
       return float(year)
   else:
       return np.nan

def process(quest, out_file):
   results = pd.DataFrame(columns=cols)
   for proj, filename in datasets.items():
      print("Processing: ", proj, " FILE: ", filename)
      if filename:
         df = pd.read_csv("data/" + proj + "/" + filename)
         quest_file = "data/" + proj + "/" + proj + quest
         question = ""
         question = Path(quest_file).read_text()
         def text_gen(x):
            return str(x['Title']) + " " + str(x['Abstract'])
         df['Text'] = df.apply(text_gen, axis=1)
         df['Project'] = proj
         df['Year'] = df['Year'].apply(coerce_year)
         search_year = np.max(df['Year'])
         df['Age'] = search_year - df['Year']
         df['Question'] = question
         if "Authors" not in df.columns:
           df["Authors"] = df["Author"]

         rez = df.loc[:,cols].copy()
         results = results.append(rez, ignore_index=True)

   results.to_csv(out_file, index=False, header=True)



###################################################################
def main():
    desc = 'join data'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('question_file',
                       metavar='question_file',
                       type=str,
                       help='The suffix of the question file')

    parser.add_argument('output_file',
                       metavar='output_file',
                       type=str,
                       help='Path to write out the results')
    args = parser.parse_args()
    quest = args.question_file
    out_file = args.output_file
    process(quest, out_file)

#########################################################################
if __name__ == '__main__':
    main()


