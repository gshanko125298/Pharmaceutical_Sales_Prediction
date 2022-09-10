import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomMaxImputer(BaseEstimator,TransformerMixin):

    def fit(self, X, y=0):
        self.fill_value  = X.max()
        return self
    def transform(self, X,y=0):
        return np.where(X.isna(), self.fill_value, X)