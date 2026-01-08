# data_generator.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_ipl_data(num_matches=5000):
    """Generate synthetic IPL match data for training"""
    
    # IPL teams
    teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore', 
             'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings', 
             'Rajasthan Royals', 'Sunrisers Hyderabad', 'Gujarat Titans', 
             'Lucknow Super Giants']
    
    # Venues
    venues = ['Wankhede Stadium', 'M. A. Chidambaram Stadium', 'Eden Gardens', 
              'Feroz Shah Kotla', 'M. Chinnaswamy Stadium', 'Sawai Mansingh Stadium',
              'Rajiv Gandhi International Stadium', 'Punjab Cricket Association Stadium',
              'Narendra Modi Stadium', 'Ekana Cricket Stadium']
    
    # Generate data
    data = []
    
    for i in range(num_matches):
        # Basic match info
        batting_team = random.choice(teams)
        bowling_team = random.choice([t for t in teams if t != batting_team])
        venue = random.choice(venues)
        
        # Match conditions
        toss_winner = random.choice([batting_team, bowling_team])
        toss_decision = random.choice(['bat', 'field'])
        
        # Overs and balls
        overs = random.randint(5, 20)
        balls = random.randint(0, 5) if overs < 20 else 0
        total_balls = overs * 6 + balls
        
        # Current score factors
        wickets = random.randint(0, 10)
        
        # Realistic score generation based on overs
        if overs <= 6:  # Powerplay
            avg_rpo = random.uniform(6, 9)
        elif overs <= 15:  # Middle overs
            avg_rpo = random.uniform(7, 10)
        else:  # Death overs
            avg_rpo = random.uniform(8, 12)
        
        current_score = int(avg_rpo * (total_balls / 6))
        
        # Adjust score based on wickets
        if wickets > 7:
            current_score = int(current_score * 0.8)
        elif wickets > 5:
            current_score = int(current_score * 0.9)
        
        # Generate final score (target)
        remaining_overs = 20 - overs - (balls / 6)
        remaining_balls = int(remaining_overs * 6)
        
        if remaining_balls > 0:
            if wickets < 3:
                final_rpo = random.uniform(8, 12)
            elif wickets < 6:
                final_rpo = random.uniform(6, 10)
            else:
                final_rpo = random.uniform(4, 8)
            
            additional_runs = int(final_rpo * remaining_overs)
            final_score = current_score + additional_runs
        else:
            final_score = current_score
        
        # Ensure realistic score range
        final_score = max(50, min(final_score, 250))
        
        # Recent form (last 5 matches performance)
        batting_team_form = random.uniform(0.3, 0.9)
        bowling_team_form = random.uniform(0.3, 0.9)
        
        # Head to head record
        head_to_head = random.uniform(0.2, 0.8)
        
        # Player impact scores
        key_batsman_form = random.uniform(0.4, 0.95)
        key_bowler_form = random.uniform(0.4, 0.95)
        
        # Venue factor
        venue_factor = random.uniform(0.7, 1.3)
        
        data.append({
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'venue': venue,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'overs': overs,
            'balls': balls,
            'current_score': current_score,
            'wickets': wickets,
            'batting_team_form': batting_team_form,
            'bowling_team_form': bowling_team_form,
            'head_to_head': head_to_head,
            'key_batsman_form': key_batsman_form,
            'key_bowler_form': key_bowler_form,
            'venue_factor': venue_factor,
            'final_score': final_score
        })
    
    return pd.DataFrame(data)

def save_data():
    """Generate and save IPL data to CSV"""
    print("Generating IPL training data...")
    df = generate_ipl_data(5000)
    
    # Save to CSV
    df.to_csv('ipl_training_data.csv', index=False)
    print(f"Data saved to ipl_training_data.csv")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Display sample data
    print("\nSample data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    save_data()