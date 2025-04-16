import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class EnsembleModel:
    def __init__(self):
        # Initialize the model and scaler
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def preprocess_data(self, df):
        """
        Preprocesses the input dataframe by encoding categorical features and scaling numerical features.
        
        Parameters:
        df (DataFrame): DataFrame containing the features and target
        
        Returns:
        X (DataFrame): Processed feature matrix
        y (Series): Target labels
        """
        # Encode categorical features using LabelEncoder
        df['hmm_state'] = self.label_encoder.fit_transform(df['hmm_state'])  # Encode HMM state
        df['lstm_pred'] = self.label_encoder.fit_transform(df['lstm_pred'])  # Encode LSTM prediction

        # Scale numerical features (e.g., sentiment score)
        df[['sentiment_score']] = self.scaler.fit_transform(df[['sentiment_score']])

        # Define features and target
        features = ['hmm_state', 'sentiment_score', 'lstm_pred']
        X = df[features]
        y = df['label']  # The target variable is the action: Long, Short, or Neutral
        
        return X, y

    def train(self, df):
        """
        Train the ensemble model on the provided dataset.

        Parameters:
        df (DataFrame): The dataset containing features and target labels
        """
        X, y = self.preprocess_data(df)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train the Random Forest model
        self.clf.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.clf.predict(X_test)
        print("Classification Report:\n", classification_report(y_test, y_pred))

        self.is_trained = True

    def predict(self, X):
        """
        Predict the market behavior (Long, Short, Neutral) using the trained model.
        
        Parameters:
        X (DataFrame): Feature matrix
        
        Returns:
        predictions (array): Array of predicted actions (Long, Short, Neutral)
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet. Please train the model first.")
        
        # Make predictions
        predictions = self.clf.predict(X)
        return predictions

    def feature_importance(self):
        """
        Plot the feature importance of the trained model.
        """
        if not self.is_trained:
            raise Exception("Model is not trained yet. Please train the model first.")
        
        # Get feature importances from the trained model
        feat_imp = pd.Series(self.clf.feature_importances_, index=['HMM State', 'Sentiment Score', 'LSTM Prediction'])
        
        # Plot feature importance
        sns.barplot(x=feat_imp, y=feat_imp.index)
        plt.title("Feature Importance")
        plt.show()


# Example Usage
if __name__ == "__main__":
    # Create a sample dataset
    data = pd.DataFrame({
        "hmm_state": ["Accumulation", "Distribution", "Distribution", "Accumulation"],
        "sentiment_score": [0.7, -0.3, -0.6, 0.9],  # Example sentiment score (positive or negative)
        "lstm_pred": ["Up", "Down", "Down", "Up"],  # LSTM predictions (Up, Down, Neutral)
        "label": ["Long", "Short", "Short", "Long"]  # Target label (Long, Short, Neutral)
    })

    # Instantiate the ensemble model
    ensemble_model = EnsembleModel()

    # Train the model with the dataset
    ensemble_model.train(data)

    # Predict on new data (Example)
    new_data = pd.DataFrame({
        "hmm_state": ["Accumulation", "Distribution"],
        "sentiment_score": [0.85, -0.4],
        "lstm_pred": ["Up", "Down"]
    })
    X_new, _ = ensemble_model.preprocess_data(new_data)
    predictions = ensemble_model.predict(X_new)
    print("Predictions on new data:", predictions)

    # Visualize feature importance
    ensemble_model.feature_importance()
