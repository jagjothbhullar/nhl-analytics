"""
NHL Player Value Analysis - Data Fetching Module
Fetches player statistics and salary data for the last 5 NHL seasons
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime

# NHL API Base URL
NHL_API_BASE = "https://api-web.nhle.com/v1"

# Seasons to analyze (last 5 full seasons)
SEASONS = ["20232024", "20222023", "20212022", "20202021", "20192020"]

def get_all_teams():
    """Fetch all NHL teams"""
    url = f"{NHL_API_BASE}/standings/now"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        teams = []
        for record in data.get('standings', []):
            teams.append({
                'team_abbrev': record['teamAbbrev']['default'],
                'team_name': record['teamName']['default'],
                'team_logo': record.get('teamLogo', '')
            })
        return teams
    return []

def get_skater_stats(season, game_type=2):
    """
    Fetch skater statistics for a season
    game_type: 2 = regular season, 3 = playoffs
    """
    url = f"{NHL_API_BASE}/skater-stats-leaders/{season}/{game_type}"

    # Get leaders in different categories to build full player list
    categories = ['goals', 'assists', 'points', 'plusMinus', 'penaltyMinutes']
    all_players = {}

    for category in categories:
        try:
            cat_url = f"{url}?categories={category}&limit=500"
            response = requests.get(cat_url)
            if response.status_code == 200:
                data = response.json()
                for cat_data in data.values():
                    if isinstance(cat_data, list):
                        for player in cat_data:
                            player_id = player.get('playerId')
                            if player_id and player_id not in all_players:
                                all_players[player_id] = {
                                    'player_id': player_id,
                                    'name': player.get('firstName', {}).get('default', '') + ' ' + player.get('lastName', {}).get('default', ''),
                                    'team': player.get('teamAbbrev', ''),
                                    'position': player.get('positionCode', ''),
                                    'games_played': player.get('gamesPlayed', 0),
                                    'goals': player.get('goals', 0),
                                    'assists': player.get('assists', 0),
                                    'points': player.get('points', 0),
                                    'plus_minus': player.get('plusMinus', 0),
                                    'season': season
                                }
            time.sleep(0.2)  # Rate limiting
        except Exception as e:
            print(f"Error fetching {category} for {season}: {e}")

    return list(all_players.values())

def get_player_details(player_id):
    """Fetch detailed player info including contract"""
    url = f"{NHL_API_BASE}/player/{player_id}/landing"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                'player_id': player_id,
                'birth_date': data.get('birthDate', ''),
                'height_inches': data.get('heightInInches', 0),
                'weight_lbs': data.get('weightInPounds', 0),
                'shoots_catches': data.get('shootsCatches', ''),
                'draft_year': data.get('draftDetails', {}).get('year', 0),
                'draft_round': data.get('draftDetails', {}).get('round', 0),
                'draft_pick': data.get('draftDetails', {}).get('pickInRound', 0),
            }
    except Exception as e:
        print(f"Error fetching player {player_id}: {e}")
    return None

def get_advanced_stats_from_nst():
    """
    Note: Natural Stat Trick requires manual CSV download
    This function provides the URL and instructions
    """
    print("\n=== Advanced Stats Instructions ===")
    print("For advanced stats (Corsi, Fenwick, xG), download from:")
    print("https://www.naturalstattrick.com/playerteams.php")
    print("Select: 5v5, All Situations, per season")
    print("Export as CSV and save as 'nst_stats.csv'")
    print("=====================================\n")

def fetch_salary_data():
    """
    Fetch salary data from PuckPedia or similar source
    Note: PuckPedia requires API access, so we'll use a backup approach
    """
    # Try to get cap hit data from NHL API player pages
    print("Fetching salary data...")
    # This would require scraping PuckPedia or using their API
    # For now, we'll create a placeholder
    return None

def main():
    print("=" * 60)
    print("NHL Player Value Analysis - Data Collection")
    print("=" * 60)

    all_stats = []

    # Fetch stats for each season
    for season in SEASONS:
        print(f"\nFetching {season[:4]}-{season[4:]} season stats...")
        stats = get_skater_stats(season)
        all_stats.extend(stats)
        print(f"  Found {len(stats)} players")
        time.sleep(1)  # Rate limiting between seasons

    # Create DataFrame
    df = pd.DataFrame(all_stats)

    # Save to CSV
    df.to_csv('nhl_player_stats.csv', index=False)
    print(f"\nSaved {len(df)} player-season records to nhl_player_stats.csv")

    # Print summary
    print("\n=== Data Summary ===")
    print(f"Total records: {len(df)}")
    print(f"Unique players: {df['player_id'].nunique()}")
    print(f"Seasons covered: {df['season'].unique()}")

    # Get advanced stats instructions
    get_advanced_stats_from_nst()

    return df

if __name__ == "__main__":
    main()
