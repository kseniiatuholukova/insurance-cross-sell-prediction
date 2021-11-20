from sklearn.ensemble import RandomForestClassifier as RFC


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return RFC().fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
