import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import matplotlib.pyplot as plt

class IPLScorePredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the IPL data"""
        print("🔄 Loading data...")
        df = pd.read_csv(csv_path)
        
        # Features to encode
        categorical_features = ['batting_team', 'bowling_team', 'venue', 'toss_winner', 'toss_decision']
        
        # Encode categorical features
        for feature in categorical_features:
            le = LabelEncoder()
            df[feature + '_encoded'] = le.fit_transform(df[feature])
            self.label_encoders[feature] = le
        
        # Select features for training
        feature_cols = [
            'batting_team_encoded', 'bowling_team_encoded', 'venue_encoded',
            'toss_winner_encoded', 'toss_decision_encoded', 'overs', 'balls',
            'current_score', 'wickets', 'batting_team_form', 'bowling_team_form',
            'head_to_head', 'key_batsman_form', 'key_bowler_form', 'venue_factor'
        ]
        
        self.feature_columns = feature_cols
        X = df[feature_cols]
        y = df['final_score']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✅ Data shape: {X.shape}")
        print(f"✅ Target range: {y.min()} - {y.max()}")
        
        return X_scaled, y, df
    
    def build_model(self, input_shape):
        """Build the deep learning model"""
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Regression output
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',   # ✅ Full loss name
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, csv_path):
        """Train the model on IPL data"""
        # Load and preprocess data
        X, y, df = self.load_and_preprocess_data(csv_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Build model
        self.model = self.build_model(X.shape[1])
        
        print("\n📐 Model Architecture:")
        self.model.summary()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001
        )
        
        # Train model
        print("\n🚀 Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        print("\n📊 Evaluating model...")
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\n✅ Training Results:")
        print(f"Train MAE: {train_mae:.2f}")
        print(f"Test MAE: {test_mae:.2f}")
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        # Save model and encoders
        self.save_model()
        
        # Plot training history
        self.plot_training_history(history)
        
        return history, (X_test, y_test, test_pred)
    
    def save_model(self):
        """Save the trained model and preprocessors"""
        print("\n💾 Saving model and preprocessors...")
        
        # Save Keras model
        self.model.save('ipl_score_predictor.h5')  # Can also use directory format
        
        # Save preprocessors
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        joblib.dump(self.feature_columns, 'feature_columns.pkl')
        
        print("✅ Model saved as 'ipl_score_predictor.h5'")
        print("✅ Preprocessors saved as 'scaler.pkl', 'label_encoders.pkl', 'feature_columns.pkl'")
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def main():
    """Main training function"""
    predictor = IPLScorePredictor()
    
    try:
        history, test_results = predictor.train_model('ipl_training_data.csv')
        print("\n🎉 Training completed successfully!")
        
    except FileNotFoundError:
        print("❌ Error: 'ipl_training_data.csv' not found!")
        print("💡 Tip: Run `data_generator.py` to generate the dataset.")
    except Exception as e:
        print(f"❌ Error during training: {e}")

if __name__ == "__main__":
    main()
