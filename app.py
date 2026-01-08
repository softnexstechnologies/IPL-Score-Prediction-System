# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
from datetime import datetime

class IPLScorePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None

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
            return None

        try:
            X_scaled = self.preprocess_input(data)
            predictions = self.model.predict(X_scaled)
            return predictions.flatten()
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            return None

# ✅ Load model and preprocessing files with caching
@st.cache_resource
def load_model_resources():
    model = load_model('ipl_score_predictor.h5')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, scaler, label_encoders, feature_columns

def main():
    st.set_page_config(page_title="IPL Score Predictor", page_icon="🏏", layout="wide")
    st.title("🏏 IPL Score Prediction -SOFTnexs Technologies")
    st.markdown("---")

    predictor = IPLScorePredictor()

    try:
        predictor.model, predictor.scaler, predictor.label_encoders, predictor.feature_columns = load_model_resources()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Info", "About"])

    if page == "Prediction":
        prediction_page(predictor)
    elif page == "Model Info":
        model_info_page()
    else:
        about_page()

def prediction_page(predictor):
    st.header("🎯 Score Prediction")

    teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
             'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
             'Rajasthan Royals', 'Sunrisers Hyderabad', 'Gujarat Titans',
             'Lucknow Super Giants']

    venues = ['Wankhede Stadium', 'M. A. Chidambaram Stadium', 'Eden Gardens',
              'Feroz Shah Kotla', 'M. Chinnaswamy Stadium', 'Sawai Mansingh Stadium',
              'Rajiv Gandhi International Stadium', 'Punjab Cricket Association Stadium',
              'Narendra Modi Stadium', 'Ekana Cricket Stadium']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Match Details")
        batting_team = st.selectbox("Batting Team", teams, index=0)
        bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team], index=0)
        venue = st.selectbox("Venue", venues, index=0)
        toss_winner = st.selectbox("Toss Winner", [batting_team, bowling_team], index=0)
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"], index=0)

    with col2:
        st.subheader("Current Match State")
        overs = st.slider("Overs Completed", 0, 20, 10)
        balls = st.slider("Balls in Current Over", 0, 5, 0)
        current_score = st.number_input("Current Score", min_value=0, max_value=300, value=80)
        wickets = st.slider("Wickets Lost", 0, 10, 2)

    st.subheader("Team & Player Form")

    col3, col4 = st.columns(2)

    with col3:
        batting_team_form = st.slider("Batting Team Form", 0.0, 1.0, 0.75, step=0.05)
        bowling_team_form = st.slider("Bowling Team Form", 0.0, 1.0, 0.65, step=0.05)
        head_to_head = st.slider("Head to Head (Batting Team)", 0.0, 1.0, 0.5, step=0.05)

    with col4:
        key_batsman_form = st.slider("Key Batsman Form", 0.0, 1.0, 0.8, step=0.05)
        key_bowler_form = st.slider("Key Bowler Form", 0.0, 1.0, 0.7, step=0.05)
        venue_factor = st.slider("Venue Factor", 0.5, 1.5, 1.0, step=0.1)

    if st.button("🎯 Predict Final Score", type="primary"):
        input_data = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'venue': [venue],
            'toss_winner': [toss_winner],
            'toss_decision': [toss_decision],
            'overs': [overs],
            'balls': [balls],
            'current_score': [current_score],
            'wickets': [wickets],
            'batting_team_form': [batting_team_form],
            'bowling_team_form': [bowling_team_form],
            'head_to_head': [head_to_head],
            'key_batsman_form': [key_batsman_form],
            'key_bowler_form': [key_bowler_form],
            'venue_factor': [venue_factor]
        })

        prediction = predictor.predict(input_data)

        if prediction is not None:
            predicted_score = int(prediction[0])
            st.markdown("---")
            st.subheader("📊 Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Predicted Final Score", f"{predicted_score}", f"{predicted_score - current_score} runs to go")

            with col2:
                remaining_overs = 20 - overs - (balls / 6)
                if remaining_overs > 0:
                    required_rpo = (predicted_score - current_score) / remaining_overs
                    st.metric("Required Run Rate", f"{required_rpo:.2f}")
                else:
                    st.metric("Match Status", "Completed")

            with col3:
                current_rpo = current_score / (overs + balls / 6) if (overs + balls / 6) > 0 else 0
                st.metric("Current Run Rate", f"{current_rpo:.2f}")

            st.subheader("📈 Score Projection")
            overs_range = list(range(0, 21))
            projected_scores = []

            for over in overs_range:
                if over <= overs:
                    projected_scores.append(int(current_score * over / max(overs, 1)))
                else:
                    remaining = 20 - over
                    if remaining > 0:
                        projected = predicted_score - ((predicted_score - current_score) * remaining / (20 - overs))
                        projected_scores.append(int(projected))
                    else:
                        projected_scores.append(predicted_score)

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=overs_range[:overs+1], y=projected_scores[:overs+1], mode='lines+markers', name='Actual Score', line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=overs_range[overs:], y=projected_scores[overs:], mode='lines+markers', name='Projected Score', line=dict(color='red', width=3, dash='dash')))
            fig.add_trace(go.Scatter(x=[overs], y=[current_score], mode='markers', name='Current Position', marker=dict(color='green', size=12)))

            fig.update_layout(title='Score Progression', xaxis_title='Overs', yaxis_title='Score', height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("💡 Insights")
            col1, col2 = st.columns(2)

            with col1:
                st.info(f"**Score Range**: {predicted_score - 15} - {predicted_score + 15}")
                st.info(f"**Wickets in Hand**: {10 - wickets}")

            with col2:
                if predicted_score > 180:
                    st.success("High scoring prediction!")
                elif predicted_score < 120:
                    st.warning("Low scoring prediction")
                else:
                    st.info("Moderate scoring prediction")

                if wickets > 6:
                    st.warning("Too many wickets lost - difficult to accelerate")
                elif wickets < 3:
                    st.success("Good platform set - can accelerate")

def model_info_page():
    st.header("🤖 Model Information")
    st.markdown("""### Model Architecture...
    (same as your original content — keep this part unchanged)
    """)

def about_page():
    st.header("ℹ️ About IPL Score Predictor")
    st.markdown("""### Overview...
    (same as your original content — keep this part unchanged)
    """)

if __name__ == "__main__":
    main()
