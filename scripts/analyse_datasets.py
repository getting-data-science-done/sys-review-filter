import pandas as pd
import numpy as np
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

descrip={
   "PID":         "Primary Immunodeficiency",
   "PBRT":        "Partial Breast Radiotherapy",
   "CIDP":        "Neurological Inflammation",
   "ESG":         "Endoscopic Sleeve Gastroplasty",
   "EMR":         "Endoscopic Mucosal Resection",
   "IORT":        "Intraoperative Radiotherapy",
   "CARGEL":      "Ligament Substitution",
   "LMTA":        "Lung Microwave Tissue Ablation",
   "LANB":        "Post-surgical Nerve Blockade",
   "VERT":        "Oesteoporotic Vertebral Fractures"
}

results = pd.DataFrame(columns=["Project","Description","Records","Relevant", "Incl Rate"])

for proj, filename in datasets.items():
   print("Processing: ", proj, " FILE: ", filename)
   if filename:
       df = pd.read_csv("data/" + proj + "/" + filename)
       print("   RECORDS\t", len(df))
       print("   ACCEPTED\t", len(df[df['Decision']==1]))
       record = {
           "Project":proj, "Description":descrip[proj], 
           "Records":len(df), 
           "Relevant":len(df[df['Decision']==1]),
           "Incl Rate": str( round(100*len(df[df['Decision']==1])/len(df),1) ) + "%"
       }
       results = results.append(record, ignore_index=True)

print(results)

results.to_latex("results/datasets.tex", index=False, header=True, caption="Medical Research Questions Datasets", label="tab:datasets")

#   USED THIS TO DUMP OUT THE INCLUDED TITLES
#   filtered = df[ df['Decision']==1]
#   titles = filtered['Title']
#   for f in titles:
#      print("   ", f)
 

