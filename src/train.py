import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from src.preprocess import build_training_data, FEATURE_COLS
from src.player_features import load_players


def train(data_path='data/all_matches.csv', model_path='model.pkl', n_estimators=100, n=5):
    if not os.path.exists(data_path):
        data_path = 'data/results.csv'
    df = pd.read_csv(data_path)
    players_df = load_players()
    training_df, le = build_training_data(df, n=n, players_df=players_df)

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
    print(f"Test accuracy (result): {accuracy:.3f}")

    def _train_binary(y_col):
        y = training_df[y_col]
        clf_b = XGBClassifier(n_estimators=n_estimators, random_state=42, eval_metric='logloss')
        clf_b.fit(X_train, y.iloc[:split])
        acc = clf_b.score(X_test, y.iloc[split:])
        print(f"Test accuracy ({y_col}): {acc:.3f}")
        return clf_b

    clf_ou15 = _train_binary('over_1_5')
    clf_ou   = _train_binary('over_2_5')
    clf_ou35 = _train_binary('over_3_5')
    clf_btts = _train_binary('btts')

    joblib.dump({
        'model': clf,
        'feature_cols': FEATURE_COLS,
        'n': n,
        'league_encoder': le,
        'result_encoder': result_encoder,
        'players_df': players_df,
        'model_ou':   clf_ou,
        'model_ou15': clf_ou15,
        'model_ou35': clf_ou35,
        'model_btts': clf_btts,
    }, model_path)
    print(f"Model saved to {model_path}")
    return clf, accuracy


if __name__ == '__main__':
    train()
