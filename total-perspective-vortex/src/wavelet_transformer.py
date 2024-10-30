import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin


# Discrete Wavelet Transform (DWT)
#
#  Permet de conserver uniquement les coefficients importants, pour réduire la taille des données tout en gardant l'essentiel et
#  réduire le bruit pour améliorer la qualité des signaux.
#
class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db4', level=3):
        # Type de Wavelet à utiliser
        self.wavelet = wavelet

        # Nombre de niveaux de décompositions
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #n_samples  : nombre d'échantillons (epochs)
        #n_channels : nombre de canaux EEG
        #n_times    : nombre de points temporels dans chaque échantillon
        n_samples, n_channels, n_times = X.shape

        coeffs_list = []

        for i in range(n_samples):
            sample_coeffs = []
            for ch in range(n_channels):
                # Listes des coefficients pour chaque niveau
                coeffs = pywt.wavedec(X[i, ch, :], self.wavelet, level=self.level)

                rec_signals = []
                for l in range(1, self.level + 1):
                    coeffs_level = [np.zeros_like(c) if idx != l else c for idx, c in enumerate(coeffs)]

                    # Reconstruction des signaux
                    rec_signal = pywt.waverec(coeffs_level, self.wavelet)

                    # Ajustement de la longueur des signaux reconstruit
                    rec_signals.append(rec_signal[:n_times])

                sample_coeffs.append(rec_signals)

            # Shape: (n_channels, n_levels, n_times)
            sample_coeffs = np.array(sample_coeffs)
            coeffs_list.append(sample_coeffs)

        # Shape: (n_samples, n_channels, n_levels, n_times)
        X_wavelet = np.array(coeffs_list)
        X_wavelet = X_wavelet.transpose(0, 2, 1, 3)

        return X_wavelet