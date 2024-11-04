import time
import pickle

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

def train_and_check_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initiate model
    model = XGBRegressor(max_depth=4, min_child_weight=2, n_estimators=160)
    model.fit(X_train, y_train)

    # Score model
    print(f"Train score is {model.score(X_train, y_train)}")
    print(f"Test Score is {model.score(X_test, y_test)}")
    print("---------------------------")

    # Metrics
    y_preds = model.predict(X_test)
    print(f"The mean absolute error of our model is {mean_absolute_error(y_test, y_preds)}.")
    print(f"The R2 score of our model is {r2_score(y_test, y_preds)}")
    print(f"The mean squared error of our model is {mean_squared_error(y_test, y_preds)}")

    return model


def save_model(model, time):
    file_name = f"../saved_models/model_{time}.pkl"

    pickle.dump(model, open(file_name, "wb"))

    print("Model saved")

def load_model(time):
    file_name = f"../saved_models/model_{time}.pkl"

    return pickle.load(open(file_name, "rb"))
