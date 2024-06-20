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
import src.explain as xpl


experiment_name = "01  : ExtaTrees" 
target = "Decision"
prediction = "predicted"

project = pit.projit_load()
exec_id = project.start_experiment(experiment_name, sys.argv[0], params={})

df = pd.read_csv( project.get_dataset("train"), low_memory=False )

studies = list(df['Project'].unique())

features = ['Age','title_length','abstract_length','TEXT_length','title_wc','title_max_wl','title_avg_wl','title_cwd','abstract_wc','abstract_max_wl','abstract_avg_wl','abstract_cwd','TEXT_wc','TEXT_max_wl','TEXT_avg_wl','TEXT_cwd','author_count','Title_match_prop','Title_jd_','Title_ld_','Title_ji_','Title_sd_','Abstract_match_prop','Abstract_jd_','Abstract_ld_','Abstract_ji_','Abstract_sd_','TEXT_match_prop','TEXT_jd_','TEXT_ld_','TEXT_ji_','TEXT_sd_']

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
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ExtraTreesClassifier() )
    ])

    model.fit(x_train, y_train)

    #explainer = xpl.PipelineExplainer(features, model, x_train, method='proba')
    #shap_plot = "results/FeatureImportance_%s.png" % study
    #print("Creating:", shap_plot)
    #explainer.plot_shap_values(x_test, shap_plot, 100)

    test_df[prediction] = model.predict_proba(x_test)[:,1]

    metrics = eval.calc_model_metrics(test_df, target, prediction)

    for key in metrics:
        project.add_result(experiment_name, key, metrics[key], study)

project.end_experiment(experiment_name, exec_id)


