import numpy as np
from sklearn.preprocessing import StandardScaler


class HybridFeatureBuilder:
    def __init__(self, alpha=0.7):
        """
        alpha: mức ưu tiên deep feature
        """
        self.alpha = alpha
        self.scaler_feat = StandardScaler()
        self.scaler_deep = StandardScaler()
        self.is_fitted = False

    def fit(self, handcrafted, deep):
        """
        Fit scaler
        """
        self.scaler_feat.fit(handcrafted)
        self.scaler_deep.fit(deep)
        self.is_fitted = True

    def transform(self, handcrafted, deep):
        """
        Transform + fusion
        """
        if not self.is_fitted:
            raise ValueError("HybridFeatureBuilder chưa fit")

        # normalize
        feat_scaled = self.scaler_feat.transform(handcrafted)
        deep_scaled = self.scaler_deep.transform(deep)

        # weighting
        feat_scaled = feat_scaled * (1 - self.alpha)
        deep_scaled = deep_scaled * self.alpha

        # concat
        combined = np.concatenate([feat_scaled, deep_scaled], axis=1)

        return combined

    def fit_transform(self, handcrafted, deep):
        self.fit(handcrafted, deep)
        return self.transform(handcrafted, deep)