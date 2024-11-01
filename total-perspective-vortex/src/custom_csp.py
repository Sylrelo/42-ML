

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh

# Common Spatial Patterns
# Extraction des caractéristiques maximisant la séparation entre deux classes de signaux
class CustomCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, tikhonov_epsilon=1e-6):
        self.n_components = n_components
        self.tikhonov_epsilon = tikhonov_epsilon
        
        self._filters = None
        
        if self.n_components is None:
            self.n_components = 4
            
        self._cache_unique_vals = None

    def fit(self, X, y):
        
        # CSP fonctionne uniquement avec deux classes.
        # Si > 2 classes, on peux utiliser du One-vs-One ou One-vs-All (comme dans DSLR eeeeh)
        if self._cache_unique_vals is None:
            self._cache_unique_vals = np.unique(y)
            assert len(self._cache_unique_vals) == 2
            
        class_1 = X[y == self._cache_unique_vals[0]]
        class_2 = X[y == self._cache_unique_vals[1]]
        
        # Calcul des matrices de covariances et des moyennes pour avoir une matrice contenant la covariance moyenne
        # Covariance: Indique le degré de variation conjointe de deux variables
        # -> Si la covariance entre deux variables est positive, elles augmentent ou diminuent ensemble
        # -> Si elle est négative, quand l'une augmente, l'autre diminue
        cov_class_1 = np.mean([np.cov(trial) for trial in class_1], axis=0)
        cov_class_2 = np.mean([np.cov(trial) for trial in class_2], axis=0)
  
        assert np.allclose(cov_class_1, cov_class_1.T)
        assert np.allclose(cov_class_2, cov_class_2.T)
        
        # Tikhonov Regularization (Ridge Regression)
        # Permet de résoudre l'instabilité des résultats sur des données bruitée ou mal processée en ajoutant un biais
        B_reg = cov_class_1 + cov_class_2 + self.tikhonov_epsilon * np.eye(cov_class_1.shape[0])

        # Décomposition en valeurs propre de la matrice de covariance
        # Représente la "direction" de la séparation des classes
        # -> Une valeur élevée signifie une grande variance entre les deux classes
        # -> Les directions élevée "extrême" (les plus grandes vs les plus petites) permettent de différencier les signaux des deux classes
        eigvals, eigvecs = eigh(cov_class_1, B_reg)
        sorted_indices = np.argsort(eigvals)[::-1]
        
        # On sélectionn les N premiers correspondant aux directions maximisant la séparation des classes
        self._filters = eigvecs[:, sorted_indices[:self.n_components]]
   
        return self

    def transform(self, X):
        # Pour chaque EPOCH, appliquer la transformation CSP pour en extraire les valeurs importantes
        transformed_x = np.array([np.dot(self._filters.T, epoch) for epoch in X])
        # Calcul de la variance des valeurs dans le temps pour chaque essai (et la "normaliser" avec le logarithme)
        transformed_x = np.log(np.var(transformed_x, axis=2))
        return transformed_x

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

