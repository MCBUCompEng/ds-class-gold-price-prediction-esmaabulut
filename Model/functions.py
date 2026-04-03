import numpy as np
import pandas as pd


#  CLEAN NUMBER

def clean_number(x):
    if isinstance(x, str):
        x = x.replace('.', '')
        x = x.replace(',', '.')
    try:
        return float(x)
    except:
        return np.nan


# LOAD + PREPROCESS (FIXED)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep=';', encoding='latin-1')

    df.columns = df.columns.str.strip()

    
    if len(df.columns) == 1:
        df = df.iloc[:, 0].str.split(';', expand=True)
        df.columns = ['Tarih', 'Acilis', 'Kapanis']

    # numeric fix
    df['Acilis'] = df['Acilis'].astype(str).apply(clean_number)
    df['Kapanis'] = df['Kapanis'].astype(str).apply(clean_number)

    # tarih fix (KRİTİK)
    df['Tarih'] = pd.to_datetime(df['Tarih'], errors='coerce', dayfirst=True)

    # SADECE gerekli kolonlar
    df = df.dropna(subset=['Acilis', 'Kapanis'])

    df = df.sort_values('Tarih')

    if len(df) == 0:
        raise ValueError("Dataset boş! CSV formatını kontrol et.")

    X = df[['Acilis']].values.astype(float)
    y = df['Kapanis'].values.astype(float)

    # scaling
    X_mean = X.mean()
    X_std = X.std()

    X_scaled = (X - X_mean) / (X_std + 1e-8)

    return X_scaled, y, X_mean, X_std, df



# POLYNOMIAL FEATURES

def create_polynomial_features(X, degree):
    X_poly = X.copy()

    for d in range(2, degree + 1):
        X_poly = np.hstack((X_poly, X ** d))

    return X_poly


# LINEAR REGRESSION (FROM SCRATCH)

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

    r2 = 1 - (
        np.sum((y_test - y_pred) ** 2) /
        np.sum((y_test - np.mean(y_test)) ** 2)
    )

    return mse, r2
