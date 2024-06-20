from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import projit as pit
import sys

sys.path.insert(1, "./")
import src.eval as eval

experiment_name = "01b : ExtaTrees - TFID - PICO"
target = "Decision"
prediction = "predicted"

project = pit.projit_load()
exec_id = project.start_experiment(experiment_name, sys.argv[0], params={})

df = pd.read_csv( project.get_dataset("train_pico"), low_memory=False )

studies = list(df['Project'].unique())

features = ['Age','title_length','abstract_length','TEXT_length',
       'title_wc','title_max_wl','title_avg_wl','title_cwd',
       'abstract_wc','abstract_max_wl','abstract_avg_wl','abstract_cwd',
       'TEXT', 'TEXT_wc','TEXT_max_wl','TEXT_avg_wl','TEXT_cwd','author_count',
       'P_TEXT_match_prop', 'P_TEXT_jd', 'P_TEXT_ld',
       'P_TEXT_ji', 'P_TEXT_sd', 'I_TEXT_match_prop', 'I_TEXT_jd', 'I_TEXT_ld',
       'I_TEXT_ji', 'I_TEXT_sd', 'C_TEXT_match_prop', 'C_TEXT_jd', 'C_TEXT_ld',
       'C_TEXT_ji', 'C_TEXT_sd', 'O_TEXT_match_prop', 'O_TEXT_jd', 'O_TEXT_ld',
       'O_TEXT_ji', 'O_TEXT_sd']

for study in studies:
    print("Test Set:", study)
    test_df = df[ df['Project']==study ].copy()
    train_df = df[ df['Project']!=study ].copy()
    y_train = train_df[target].fillna(0)
    x_train = train_df.loc[:,features]

    y_test = test_df[target].fillna(0)
    x_test = test_df.loc[:,features]

    numeric_cols = list( x_train.select_dtypes(include="number").columns)

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median')
    )
    text_transformer = make_pipeline(
        TfidfVectorizer(max_features=200, stop_words='english')
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("text", text_transformer, 'TEXT'),
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ExtraTreesClassifier() )
    ])

    model.fit(x_train, y_train)

    test_df[prediction] = model.predict_proba(x_test)[:,1]

    metrics = eval.calc_model_metrics(test_df, target, prediction)

    for key in metrics:
        project.add_result(experiment_name, key, metrics[key], study)

project.end_experiment(experiment_name, exec_id)


