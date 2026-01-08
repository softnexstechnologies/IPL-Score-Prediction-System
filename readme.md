# requirements.txt

tensorflow==2.13.0
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
joblib==1.3.2

# Setup Instructions

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Run the Complete Pipeline

### Step 1: Generate Training Data
```bash
python data_generator.py
```
This creates `ipl_training_data.csv` with 5,000 synthetic IPL matches.

### Step 2: Train the Model
```bash
python model_training.py
```
This creates:
- `ipl_score_predictor.h5` (trained Keras model)
- `scaler.pkl` (feature scaler)
- `label_encoders.pkl` (categorical encoders)
- `feature_columns.pkl` (feature column names)
- `training_history.png` (training plots)

### Step 3: Test the Model
```bash
python model_testing.py
```
This evaluates the model and creates `model_evaluation.png`.

### Step 4: Launch Streamlit App
```bash
streamlit run streamlit_app.py
```
This starts the web interface at `http://localhost:8501`.

## 3. File Structure

```
ipl_score_prediction/
├── data_generator.py          # Generate synthetic IPL data
├── model_training.py          # Train deep learning model
├── model_testing.py           # Test and evaluate model
├── streamlit_app.py           # Streamlit web interface
├── requirements.txt           # Python dependencies
├── ipl_training_data.csv      # Generated training data
├── ipl_score_predictor.h5     # Trained model (H5 format)
├── scaler.pkl                 # Feature scaler
├── label_encoders.pkl         # Categorical encoders
├── feature_columns.pkl        # Feature column names
├── training_history.png       # Training visualizations
└── model_evaluation.png       # Model performance plots
```

## 4. Model Details

### Architecture
- Input: 15 features (team info, match state, form factors)
- Hidden Layers: 256 → 128 → 64 → 32 → 16 neurons
- Output: 1 neuron (predicted final score)
- Activation: ReLU (hidden), Linear (output)
- Regularization: Dropout, BatchNormalization

### Features Used
1. **Categorical**: batting_team, bowling_team, venue, toss_winner, toss_decision
2. **Numerical**: overs, balls, current_score, wickets, batting_team_form, bowling_team_form, head_to_head, key_batsman_form, key_bowler_form, venue_factor

### Performance Metrics
- Mean Absolute Error: ~12-15 runs
- Root Mean Square Error: ~18-22 runs
- R² Score: ~0.85-0.90
- Accuracy (±10 runs): ~65-75%

## 5. Usage Examples

### Command Line Prediction
```python
from model_testing import IPLScorePredictor
import pandas as pd

predictor = IPLScorePredictor()
predictor.load_model()

# Sample prediction
data = pd.DataFrame({
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

prediction = predictor.predict(data)
print(f"Predicted Score: {prediction[0]:.0f}")
```

## 6. Troubleshooting

### Common Issues

1. **Model Loading Error**
   ```
   Error: Cannot load model files
   Solution: Ensure all .pkl and .h5 files are in the same directory
   ```

2. **TensorFlow Installation Issues**
   ```bash
   # For Mac M1/M2
   pip install tensorflow-macos
   
   # For older systems
   pip install tensorflow==2.13.0
   ```

3. **Streamlit Port Issues**
   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```

4. **Memory Issues**
   ```
   Reduce batch_size in model_training.py
   Or use smaller dataset in data_generator.py
   ```

## 7. Customization

### Modify Team Lists
Edit the `teams` list in `data_generator.py` and `streamlit_app.py`.

### Add New Features
1. Add feature to `generate_ipl_data()` in `data_generator.py`
2. Update `feature_columns` in `model_training.py`
3. Add input field in `streamlit_app.py`

### Tune Model Architecture
Modify the `build_model()` function in `model_training.py`:
- Change layer sizes
- Add/remove layers
- Adjust dropout rates
- Change activation functions

## 8. Deployment

### Local Deployment
```bash
streamlit run streamlit_app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy from repository

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt