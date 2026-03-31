import numpy as np
from Model.functions import (
    load_and_preprocess_data,
    train_model,
    evaluate_model,
    create_polynomial_features
)

#  VERİYİ YÜKLE
X, y, X_mean, X_std = load_and_preprocess_data("DataSet/gram_gold_10yrs.csv")

degrees = range(1, 6)
splits = [0.7, 0.8, 0.9]

best_r2 = -999
best_model = None
best_degree = None
best_split = None

#  MODEL DENEME
for split in splits:
    print(f"\n===== TRAIN SPLIT %{int(split*100)} =====")

    for d in degrees:

        X_poly = create_polynomial_features(X, d)

        split_index = int(len(X_poly) * split)

        X_train = X_poly[:split_index]
        X_test = X_poly[split_index:]

        y_train = y[:split_index]
        y_test = y[split_index:]

        model = train_model(X_train, y_train)

        mse, r2 = evaluate_model(model, X_test, y_test)

        print(f"Degree {d} -> MSE: {mse:.2f}, R2: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_degree = d
            best_split = split

print("\n BEST MODEL")
print(f"Degree: {best_degree}")
print(f"Split: %{int(best_split*100)}")
print(f"R2: {best_r2:.4f}")

# SON GÜN TAHMİNİ
X_best = create_polynomial_features(X, best_degree)

last_point = X_best[-1].reshape(1, -1)

prediction = best_model.predict(last_point)

print("\n============================")
print("Tomorrow Prediction:", prediction[0])
print("============================")
