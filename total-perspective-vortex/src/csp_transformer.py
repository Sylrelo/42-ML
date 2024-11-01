import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from custom_csp import CustomCSP


class CSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.csp_list = []

    def fit(self, X, y):
        _, n_levels, _, _ = X.shape
        self.csp_list = []

        # Analyse CSP pour chaque bande de fréquence disponible
        for level in range(n_levels):
            csp = CustomCSP(n_components=self.n_components)
            X_level = X[:, level, :, :]
            csp.fit(X_level, y)
            self.csp_list.append(csp)

        return self

    def transform(self, X):
        X_csp = []
        for level, csp in enumerate(self.csp_list):
            X_level = X[:, level, :, :]
            X_csp_level = csp.transform(X_level)
            X_csp.append(X_csp_level)

        X_csp = np.concatenate(X_csp, axis=1)
        return X_csp