from Model.functions import (
    load_and_preprocess_data,
    train_model,
    evaluate_model,
    create_polynomial_features
)

# veri yükle
X, y = load_and_preprocess_data("DataSet/gram_gold_10yrs.csv")

# 🎯 farklı split oranları
split_ratios = [0.7, 0.8, 0.9]

# 🎯 polynomial dereceler
degrees = range(1, 7)

best_r2 = -999
best_config = None

for split in split_ratios:
    print(f"\n===== SPLIT: %{int(split*100)} TRAIN =====")

    for degree in degrees:

        X_poly = create_polynomial_features(X, degree)

        split_index = int(len(X_poly) * split)

        X_train = X_poly[:split_index]
        X_test = X_poly[split_index:]

        y_train = y[:split_index]
        y_test = y[split_index:]

        # model eğit
        w, b = train_model(X_train, y_train)

        # evaluate
        mse, r2 = evaluate_model(w, b, X_test, y_test)

        print(f"Degree {degree} -> MSE: {mse:.2f}, R2: {r2:.4f}")

        # en iyiyi seç
        if r2 > best_r2:
            best_r2 = r2
            best_config = (degree, split)

print("\n🏆 EN İYİ MODEL")
print(f"Degree: {best_config[0]}")
print(f"Train oranı: %{int(best_config[1]*100)}")
print(f"R2: {best_r2}")
