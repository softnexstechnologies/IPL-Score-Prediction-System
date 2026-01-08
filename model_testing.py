# model_testing.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class IPLScorePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        
    def load_model(self, model_path='ipl_score_predictor.h5'):
        """Load the trained model and preprocessors"""
        try:
            print("Loading model and preprocessors...")
            self.model = load_model(model_path)
            self.scaler = joblib.load('scaler.pkl')
            self.label_encoders = joblib.load('label_encoders.pkl')
            self.feature_columns = joblib.load('feature_columns.pkl')
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_input(self, data):
        """Preprocess input data for prediction"""
        df = data.copy()
        
        # Encode categorical features
        categorical_features = ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']
        
        for feature in categorical_features:
            if feature in df.columns:
                le = self.label_encoders[feature]
                df[feature + '_encoded'] = le.transform(df[feature])
        
        # Select and scale features
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, data):
        """Make predictions on new data"""
        if self.model is None:
            print("Model not loaded. Please load model first.")
            return None
        
        try:
            X_scaled = self.preprocess_input(data)
            predictions = self.model.predict(X_scaled)
            return predictions.flatten()
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def evaluate_model(self, test_csv_path):
        """Evaluate model on test data"""
        print("Evaluating model on test data...")
        
        # Load test data
        test_df = pd.read_csv(test_csv_path)
        
        # Make predictions
        predictions = self.predict(test_df)
        
        if predictions is None:
            return None
        
        # Calculate metrics
        y_true = test_df['final_score'].values
        
        mae = mean_absolute_error(y_true, predictions)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        r2 = r2_score(y_true, predictions)
        
        # Calculate accuracy within ranges
        accuracy_5 = np.mean(np.abs(y_true - predictions) <= 5) * 100
        accuracy_10 = np.mean(np.abs(y_true - predictions) <= 10) * 100
        accuracy_15 = np.mean(np.abs(y_true - predictions) <= 15) * 100
        
        print(f"\nModel Performance:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Square Error: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Accuracy within 5 runs: {accuracy_5:.1f}%")
        print(f"Accuracy within 10 runs: {accuracy_10:.1f}%")
        print(f"Accuracy within 15 runs: {accuracy_15:.1f}%")
        
        # Visualize results
        self.visualize_predictions(y_true, predictions)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy_5': accuracy_5,
            'accuracy_10': accuracy_10,
            'accuracy_15': accuracy_15
        }
    
    def visualize_predictions(self, y_true, predictions):
        """Visualize model predictions"""
        plt.figure(figsize=(15, 10))
        
        # Scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, predictions, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Score')
        plt.ylabel('Predicted Score')
        plt.title('Actual vs Predicted Scores')
        
        # Residual plot
        plt.subplot(2, 2, 2)
        residuals = y_true - predictions
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Score')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Error distribution
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        
        # Accuracy by score range
        plt.subplot(2, 2, 4)
        score_ranges = [(0, 120), (120, 150), (150, 180), (180, 220), (220, 300)]
        accuracies = []
        range_labels = []
        
        for low, high in score_ranges:
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 0:
                acc = np.mean(np.abs(y_true[mask] - predictions[mask]) <= 10) * 100
                accuracies.append(acc)
                range_labels.append(f'{low}-{high}')
        
        plt.bar(range_labels, accuracies, alpha=0.7)
        plt.xlabel('Score Range')
        plt.ylabel('Accuracy (±10 runs)')
        plt.title('Accuracy by Score Range')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png')
        plt.show()
    
    def test_single_prediction(self):
        """Test single prediction with sample data"""
        print("\nTesting single prediction...")
        
        # Sample test data
        sample_data = pd.DataFrame({
            'batting_team': ['Mumbai Indians'],
            'bowling_team': ['Chennai Super Kings'],
            'venue': ['Wankhede Stadium'],
            'toss_winner': ['Mumbai Indians'],
            'toss_decision': ['bat'],
            'overs': [15],
            'balls': [3],
            'current_score': [145],
            'wickets': [3],
            'batting_team_form': [0.75],
            'bowling_team_form': [0.65],
            'head_to_head': [0.6],
            'key_batsman_form': [0.85],
            'key_bowler_form': [0.7],
            'venue_factor': [1.1]
        })
        
        prediction = self.predict(sample_data)
        
        if prediction is not None:
            print(f"Sample Input:")
            for col, val in sample_data.iloc[0].items():
                print(f"  {col}: {val}")
            print(f"\nPredicted Final Score: {prediction[0]:.0f}")
        
        return prediction

def main():
    """Main testing function"""
    # Initialize predictor
    predictor = IPLScorePredictor()
    
    # Load model
    if not predictor.load_model():
        print("Failed to load model. Please train the model first.")
        return
    
    # Test single prediction
    predictor.test_single_prediction()
    
    # Evaluate on test data if available
    try:
        # You can create a separate test CSV or use part of training data
        # For now, we'll use the same training data for evaluation
        metrics = predictor.evaluate_model('ipl_training_data.csv')
        
        if metrics:
            print("\nModel evaluation completed successfully!")
            
    except FileNotFoundError:
        print("Test data not found. Skipping evaluation.")
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()