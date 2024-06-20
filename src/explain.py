import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PipelineExplainer():
    """
    A wrapper around SHAP explainer for working with SKLearn Pipelines
    """

    def __init__(self, feature_names, pipeline, train_x, method='predict'):
        """
        Initialise with the feature names and the pipeline to explain
        """
        self.feature_names = feature_names
        self.pipeline = pipeline
        self.shap_vals = None

        if method == "proba":
            self.predictor = self.model_predict_proba
        else:
            self.predictor = self.model_predict

        if len(train_x)>100:
            train_x = train_x.sample(n=100, replace=False)
            #USE THIS FO NP: train_x = train_x[np.random.choice(train_x.shape[0], 200, replace=False)]

        self.explainer = shap.KernelExplainer(self.predictor, train_x)

    def model_predict(self, data):
        df =  pd.DataFrame(data, columns=self.feature_names)
        return self.pipeline.predict(df)

    def model_predict_proba(self, data):
        df =  pd.DataFrame(data, columns=self.feature_names)
        return self.pipeline.predict_proba(df)[:,1]

    def compute_shap(self, data, samples=100):
        self.shap_vals = self.explainer.shap_values(data, nsamples=samples)
        return self.shap_vals

    def plot_shap_values(self, data, filename, samples=100):
        if self.shap_vals is None:
            self.shap_vals = self.explainer.shap_values(data, nsamples=samples)
        shap.summary_plot(self.shap_vals, data, show=False)
        fig = plt.gcf()
        fig.set_figheight(8)
        fig.set_figwidth(12)
        plt.savefig(filename)

    def beeswarm_plot(self, data, filename, samples=100):
        if self.shap_vals is None:
            self.shap_vals = self.explainer.shap_values(data, nsamples=samples)
        shap.plots.beeswarm(self.shap_vals, show=False)
        fig = plt.gcf()
        fig.set_figheight(8)
        fig.set_figwidth(12)
        plt.savefig(filename)


