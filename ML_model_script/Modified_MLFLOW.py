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
            

            mlflow.log_metric("mean_squared_error", error)
            print("mean_squared_error of model is ",error)
            self.logger(model,self.X_test, self.y_test,title="Baseline_"+self.name)
            return model
    def logger(self,model,X_train, y_train,title):
        pass
    def trainKFold(self,folds,params=None):
        mlflow.end_run()
        self.callAutoLog()
        with mlflow.start_run(run_name="Kfold_"+self.name):
            mlflow.log_param('data_version', self.data_version)
            
            kf=KFold(n_splits=folds, random_state=None)
            if params:
                model = self.model(**params)
            else:
                model =self.model()

            scores=[]
            for train_index, test_index in kf.split(self.X_train):
                x_fold_train, X_val, y_fold_train, y_val = self.X_train.iloc[train_index], self.X_train.iloc[test_index], self.y_train.iloc[train_index], self.y_train.iloc[test_index]
                model.fit(x_fold_train, y_fold_train)
                predict_valid=model.predict(X_val)
                valid_loss=log_loss(y_val,predict_valid)
                scores.append(valid_loss)
                mlflow.log_metric("log_loss", valid_loss)
            self.logger(model,self.X_test, self.y_test,title="Kfold_"+self.name)
            mlflow.log_metric("avergae_validation_log_loss",np.mean( scores))
            return model

    def hyperParameterTune(self, folds, search_space,params=None,loss="neg_mean_squared_error"):
        mlflow.end_run()
        self.callAutoLog()
        if params:
            model = self.model(**params)
        else:
            model =self.model()
            cvFold = KFold(n_splits=folds)
        gridSearch = GridSearchCV(estimator=model, param_grid=search_space, n_jobs=-1,  cv=cvFold, scoring=loss)
        with mlflow.start_run(run_name='hyperparam_tuning_'+self.name) as run:
            
            mlflow.log_param('data_version', self.data_version)
            searchResults = gridSearch.fit(self.X_train, self.y_train)
        
            pred=searchResults.predict(self.X_test)
            
            error=mean_squared_error(self.y_test,pred)

            mlflow.log_metric("mean_squared_error", error)
            print("mean_squared_error of model is ",error)
            
        return(searchResults.best_estimator_)
    def callAutoLog(self):
            mlflow.sklearn.autolog()
    def getFeatureImportance(self):
        feature_importances = pd.DataFrame((self.model.feature_importances_).transpose() , index=self.X_train.columns.tolist(), columns=['importance'])
        return feature_importances.sort_values('importance', ascending=False)