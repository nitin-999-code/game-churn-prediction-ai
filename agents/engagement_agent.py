"""
Engagement Agent

Analyzes player behavior and returns targeted retention strategies based on predefined rules.
"""

import json
import os
import pandas as pd

def load_strategies():
    """Loads engagement strategies from the knowledge base."""
    file_path = os.path.join(os.path.dirname(__file__), "..", "knowledge_base", "strategies.json")
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def analyze_player_behavior(player):
    """
    Analyzes a single player's properties (pandas Series or dict) and 
    returns a list of identified risk behaviors.
    """
    risk_behaviors = []
    
    # 1. Low sessions
    if player.get('SessionsPerWeek', 5) < 3:
        risk_behaviors.append("low_sessions")
        
    # 2. Short duration
    if player.get('AvgSessionDurationMinutes', 30) < 15:
        risk_behaviors.append("short_duration")
        
    # 3. Low progress (Check if achievements are low compared to expected level)
    # Defaulting if missing for safety
    achievements = player.get('AchievementsUnlocked', 10)
    level = player.get('PlayerLevel', 1)
    if achievements < (level * 0.5):
        risk_behaviors.append("low_progress")
        
    # 4. No purchases
    purchases = player.get('InGamePurchases', 1)
    if purchases == 0:
        risk_behaviors.append("no_purchases")
        
    # 5. Declining activity (Assuming PlayTimeHours vs expected)
    if player.get('PlayTimeHours', 10) < 5:
        risk_behaviors.append("declining_activity")
        
    return risk_behaviors

def handle_missing_values(player_series):
    """Fills any missing values for a player row using simple defaults."""
    if isinstance(player_series, dict):
        filled_player = player_series.copy()
        for k, v in filled_player.items():
            if pd.isna(v):
                filled_player[k] = 0
        return filled_player
        
    filled_player = player_series.copy()
    for col in filled_player.index:
        if pd.isna(filled_player[col]):
            filled_player[col] = 0
    return filled_player

def generate_recommendations(player_row):
    """
    Takes a player row, analyzes behavior, and generates recommendations.
    Returns a list of recommendation strings.
    """
    strategies_kn = load_strategies()
    player = handle_missing_values(player_row)
    behaviors = analyze_player_behavior(player)
    
    recommendations = []
    for behavior in behaviors:
        strats = strategies_kn.get(behavior, [])
        recommendations.extend(strats)
        
    # Remove duplicates but keep order
    unique_recommendations = []
    for rec in recommendations:
        if rec not in unique_recommendations:
            unique_recommendations.append(rec)
            
    # If no specific recommendations, add a general one
    if not unique_recommendations:
        unique_recommendations.append("Continue current engagement tracking.")
        
    return unique_recommendations
