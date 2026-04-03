import numpy as np
from Model.functions import (
    load_and_preprocess_data,
    create_polynomial_features,
    train_model,
    evaluate_model
)

# LOAD DATA
X, y, X_mean, X_std, df = load_and_preprocess_data(
    "DataSet/gram_gold_10yrs.csv"
)

degrees = range(1, 6)
splits = [0.7, 0.8, 0.9]

best_r2 = -999
best_model = None
best_degree = None
best_split = None


#TRAIN LOOP

for split in splits:
    print(f"\n===== TRAIN SPLIT %{int(split*100)} =====")

    split_index = int(len(X) * split)

    X_train_raw = X[:split_index]
    X_test_raw = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    for d in degrees:

        X_train = create_polynomial_features(X_train_raw, d)
        X_test = create_polynomial_features(X_test_raw, d)

        model = train_model(X_train, y_train)

        mse, r2 = evaluate_model(model, X_test, y_test)

        print(f"Degree {d} -> MSE: {mse:.2f}, R2: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_degree = d
            best_split = split


#BEST MODEL

print("\n===== BEST MODEL =====")
print(f"Degree: {best_degree}")
print(f"Split: %{int(best_split*100)}")
print(f"R2: {best_r2:.4f}")


#  NEXT DAY PREDICTION

last_point = X[-1:].copy()
last_point_poly = create_polynomial_features(last_point, best_degree)

prediction = best_model.predict(last_point_poly)

print("\n============================")
print("Next Day Gold Prediction (TL):", prediction[0])
print("============================")
