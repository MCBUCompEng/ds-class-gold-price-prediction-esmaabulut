import numpy as np
import pandas as pd


# VERİ YÜKLE + PREPROCESS
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df = df.dropna()

    X = df[['Opening TL']].values.astype(float)
    y = df['Closing TL'].values.astype(float)

    # scaling
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std

    return X_scaled, y, X_mean, X_std


# POLYNOMIAL FEATURES
def create_polynomial_features(X, degree):
    X_poly = X.copy()
    for d in range(2, degree + 1):
        X_poly = np.hstack((X_poly, X ** d))
    return X_poly


# LINEAR REGRESSION (GRADIENT DESCENT)
class LinearRegression:
    def __init__(self, lr=0.01, epochs=2000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b


# TRAIN
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# EVALUATE
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = np.mean((y_test - y_pred) ** 2)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    return mse, r2
