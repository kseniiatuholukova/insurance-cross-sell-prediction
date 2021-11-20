from sklearn.ensemble import RandomForestClassifier as RFC

from utils.dataloader import Resampler


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        sampler = Resampler()
        X_resampled, y_resampled = sampler.fit(train_x, train_y)
        return RFC().fit(X_resampled, y_resampled)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
