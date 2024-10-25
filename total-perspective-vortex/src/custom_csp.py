

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh

class CustomCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components
        self._filters = None
        
        if self.n_components is None:
            self.n_components = 4
            
        self._cache_unique_vals = None

    def fit(self, X, y):
        
        if self._cache_unique_vals is None:
            self._cache_unique_vals = np.unique(y)
            assert len(self._cache_unique_vals) == 2
            
        class_1 = X[y == self._cache_unique_vals[0]]
        class_2 = X[y == self._cache_unique_vals[1]]
        
        cov_class_1 = np.mean([np.cov(trial) for trial in class_1], axis=0)
        cov_class_2 = np.mean([np.cov(trial) for trial in class_2], axis=0)
  
        assert np.allclose(cov_class_1, cov_class_1.T)
        assert np.allclose(cov_class_2, cov_class_2.T)

        eigvals, eigvecs = eigh(cov_class_1, cov_class_1 + cov_class_2)
        sorted_indices = np.argsort(eigvals)[::-1]
        self._filters = eigvecs[:, sorted_indices[:self.n_components]]
   
        return self

    def transform(self, X):
        """Compute CSP filter on X Datas and returns the power of
            CSP features averaged over time and shape"""
        transformed_x = np.array([np.dot(self._filters.T, epoch) for epoch in X])
        transformed_x = np.log(np.var(transformed_x, axis=2))
        # transformed_x = (**2).mean(axis=2)
        return transformed_x

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    # def _get_2D_cov(self, X_class):
    #     n_channels = X_class.shape[1]
    #     X_class = np.transpose(X_class, [1, 0, 2])
    #     X_class = X_class.reshape(n_channels, -1)
    #     return np.cov(X_class)