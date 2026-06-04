import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.preprocess import build_training_data, FEATURE_COLS


def train(data_path='data/results.csv', model_path='model.pkl', n_estimators=100, n=5):
    df = pd.read_csv(data_path)
    training_df = build_training_data(df, n=n)

    X = training_df[FEATURE_COLS]
    y = training_df['result']

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.3f}")

    joblib.dump({'model': clf, 'feature_cols': FEATURE_COLS, 'n': n}, model_path)
    print(f"Model saved to {model_path}")
    return clf, accuracy


if __name__ == '__main__':
    train()
