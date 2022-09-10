from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

from sklearn.metrics import mean_squared_error, accuracy_score, log_loss

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import mlflow
import mlflow.sklearn

import numpy as np

from sklearn.model_selection import GridSearchCV
import pandas as pd

class CreateModel:
    def __init__(self, X_train, X_test, y_train, y_test,data_version,name,model):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.model=model
        self.featureImportance=[]
        self.name=name
        self.data_version=data_version
        
        
        mlflow.end_run()
        mlflow.set_experiment(self.name+"_Experiment")
    def train(self,params=None):
        mlflow.end_run()
        self.callAutoLog()
        
        with mlflow.start_run(run_name="Baseline_"+self.name):
            mlflow.log_param('data_version', self.data_version)

            if params:
                model = self.model(**params)
            else:
                model = self.model()
            model.fit(self.X_train, self.y_train)

            pred = model.predict(self.X_test)
            # print(pred)
            error=mean_squared_error(self.y_test,pred)
            