import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from src.preprocess import build_training_data, FEATURE_COLS


def train(data_path='data/all_matches.csv', model_path='model.pkl', n_estimators=100, n=5):
    if not os.path.exists(data_path):
        data_path = 'data/results.csv'
    df = pd.read_csv(data_path)
    training_df, le = build_training_data(df, n=n)

    X = training_df[FEATURE_COLS]
    y = training_df['result']

    # Encode target to integers (required by XGBoost 3.x)
    result_encoder = LabelEncoder()
    y_encoded = result_encoder.fit_transform(y)

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_encoded[:split], y_encoded[split:]

    clf = XGBClassifier(n_estimators=n_estimators, random_state=42, eval_metric='mlogloss')
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.3f}")

    joblib.dump({
        'model': clf,
        'feature_cols': FEATURE_COLS,
        'n': n,
        'league_encoder': le,
        'result_encoder': result_encoder,
    }, model_path)
    print(f"Model saved to {model_path}")
    return clf, accuracy


if __name__ == '__main__':
    train()
