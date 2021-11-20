from sklearn.ensemble import GradientBoostingClassifier as GBC


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return GBC().fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
