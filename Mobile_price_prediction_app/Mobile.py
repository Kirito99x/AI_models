import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df):
    train_df.dropna(inplace=True)
    return train_df

def split_data(train_df):
    X = train_df.drop('price_range', axis=1)
    y = train_df['price_range']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test

def build_model(X_train, y_train):
    knn_model = KNeighborsClassifier(n_neighbors=10)
    knn_model.fit(X_train, y_train)
    return knn_model

def evaluate_model(model, X_test, y_test):
    score = model.score(X_test, y_test)
    print("Model Accuracy:", score)
    predictions = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

def predict_price(model, test_df):
    test_df = test_df.drop('id', axis=1)
    predicted_prices = model.predict(test_df)
    return predicted_prices

def main():
    train_path = '/home/kirito99/Mobile_price_prediction_app/dataset/mobile_price_training_data.csv'
    test_path = input("Enter the path to the test data CSV file: ")

    train_df, test_df = load_data(train_path, test_path)
    train_df = preprocess_data(train_df)
    X_train, X_test, y_train, y_test = split_data(train_df)
    model = build_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    predicted_prices = predict_price(model, test_df)
    return predicted_prices

if __name__ == "__main__":
    predicted_prices = main()
    print("Predicted Prices:", predicted_prices)
