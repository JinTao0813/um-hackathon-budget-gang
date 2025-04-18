import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel():
    def __init__(self, df):
        self.training_df = df
        self.model = None
        self.inverse_label_map = None

    def generate_labels_from_prices(self, df, future_window=5):
        """
        Generate labels based on future price movement.
        Labels: 1 = Buy, -1 = Sell, 0 = Hold
        """
        labels = []
        for i in range(len(df) - future_window):
            current_price = df.iloc[i]['close']
            future_price = df.iloc[i + future_window]['close']
            if current_price < future_price:
                labels.append(1)
            elif current_price > future_price:
                labels.append(-1)
            else:
                labels.append(0)
        labels.extend([0] * future_window)  # padding for final rows
        print("Labels generated from prices:", labels)
        return pd.Series(labels, index=df.index)


    def train(self):
        """
        Train logistic regression model using HMM and LSTM predictions.
        """
        # Generate synthetic labels using price movement
        self.training_df["target"] = self.generate_labels_from_prices(self.training_df)

        # Prepare training features and labels
        X = self.training_df[["deepPredictor", "marketRegime"]]
        y = self.training_df["target"]

        y_raw = self.training_df["target"]
        y = y_raw.map({
            -1: 0,  # sell
            0: 1,  # hold
            1: 2   # buy
        })

        if y.isnull().any():
            raise ValueError("Target column contains unexpected values. Only 'buy' and 'sell' are allowed.")

        self.inverse_label_map = {0: 'sell', 1: 'hold', 2: 'buy'}

        # Encode categorical input features using OneHotEncoder in a pipeline
        categorical_features = ["deepPredictor", "marketRegime"]
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ]
        )

        # Define and train pipeline with Logistic Regression
        self.meta_model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))  # Ensure convergence
        ])

        # Fit model
        self.meta_model.fit(X, y)

    def predict(self, predict_df):
        """
        Predict using the trained model.
        """
        print("Predict df in Logistic Regression model", predict_df)
        X = predict_df[["deepPredictor", "marketRegime"]]
        print("X: " , X)
        if self.meta_model is None:
            raise ValueError("Model not trained.")
        prediction = self.meta_model.predict(X)
        print("Prediction: ", prediction)
        return [self.inverse_label_map[p] for p in prediction]
