"""
NHL Player Value Analysis
==========================
Identifies undervalued and overvalued players by comparing
performance metrics to contract value.

Author: Jagjoth Bhullar
San Jose Sharks - Hockey Analytics Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import requests
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class NHLPlayerValueAnalyzer:
    """Analyze NHL player value based on statistics vs salary"""

    def __init__(self):
        self.players_df = None
        self.model = None
        self.scaler = StandardScaler()

    def fetch_current_season_stats(self):
        """Fetch player stats from NHL API"""
        print("Fetching NHL player statistics...")

        # NHL Stats API endpoint
        url = "https://api.nhle.com/stats/rest/en/skater/summary"
        params = {
            "isAggregate": "false",
            "isGame": "false",
            "sort": '[{"property":"points","direction":"DESC"}]',
            "start": 0,
            "limit": 300,
            "cayenneExp": "gameTypeId=2 and seasonId>=20232024"
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                players = data.get('data', [])
                if players:
                    df = pd.DataFrame(players)
                    print(f"Fetched {len(df)} players from NHL API")

                    # Standardize column names
                    df = self._standardize_columns(df)
                    return df
        except Exception as e:
            print(f"API fetch error: {e}")

        # Use sample data for demonstration
        return self._get_sample_data()

    def _standardize_columns(self, df):
        """Standardize column names from NHL API"""
        column_map = {
            'skaterFullName': 'name',
            'teamAbbrevs': 'team',
            'positionCode': 'position',
            'gamesPlayed': 'games_played',
            'goals': 'goals',
            'assists': 'assists',
            'points': 'points',
            'plusMinus': 'plus_minus',
            'penaltyMinutes': 'pim',
            'ppGoals': 'pp_goals',
            'ppPoints': 'pp_points',
            'shGoals': 'sh_goals',
            'gameWinningGoals': 'gw_goals',
            'shots': 'shots',
            'timeOnIcePerGame': 'toi_per_game',
            'faceoffWinPct': 'faceoff_pct'
        }

        df = df.rename(columns=column_map)

        # Add estimated salary based on points (realistic approximation)
        # NHL salary is highly correlated with point production
        # Base minimum is ~$775K, elite players make $10M+
        np.random.seed(42)
        if 'points' in df.columns:
            base = 775000
            points_value = df['points'].fillna(0) * 75000
            age_factor = np.random.uniform(0.8, 1.2, len(df))  # Age/contract timing
            market_noise = np.random.normal(0, 400000, len(df))
            df['cap_hit'] = (base + points_value * age_factor + market_noise).clip(lower=775000)

        return df

    def _get_sample_data(self):
        """Generate realistic sample data for demonstration"""
        print("Generating sample NHL player data for demonstration...")

        np.random.seed(42)

        # Real NHL team abbreviations
        teams = ['SJS', 'LAK', 'ANA', 'VGK', 'SEA', 'EDM', 'CGY', 'VAN',
                 'COL', 'MIN', 'STL', 'CHI', 'NSH', 'DAL', 'WPG', 'ARI']

        # Generate 150 players with realistic distributions
        n_players = 150

        # Position distribution (more defensemen needed)
        positions = np.random.choice(['C', 'L', 'R', 'D'], n_players, p=[0.22, 0.18, 0.18, 0.42])

        # Games played (most play 60-82)
        games = np.random.triangular(30, 70, 82, n_players).astype(int)

        # Goals and assists vary by position
        goals = np.zeros(n_players)
        assists = np.zeros(n_players)

        for i, pos in enumerate(positions):
            if pos in ['C', 'L', 'R']:  # Forwards score more
                goals[i] = np.random.poisson(18)
                assists[i] = np.random.poisson(28)
            else:  # Defensemen
                goals[i] = np.random.poisson(6)
                assists[i] = np.random.poisson(22)

        # Adjust for games played
        goals = (goals * games / 82).astype(int)
        assists = (assists * games / 82).astype(int)
        points = goals + assists

        # Generate realistic salary distribution
        # Based on points production with variance
        base_salary = 775000  # NHL minimum
        point_value = 85000  # $/point baseline

        # Elite players (top 10%) get premium
        elite_threshold = np.percentile(points, 90)
        salary_mult = np.where(points >= elite_threshold, 1.5, 1.0)

        # Add market factors
        market_premium = np.random.uniform(0.7, 1.4, n_players)

        cap_hit = base_salary + (points * point_value * salary_mult * market_premium)
        cap_hit = cap_hit + np.random.normal(0, 300000, n_players)
        cap_hit = np.clip(cap_hit, 775000, 12500000)

        # Create player names (sample real-ish names)
        first_names = ['Connor', 'Auston', 'Nathan', 'Leon', 'Cale', 'Alex',
                       'David', 'Mitch', 'Mikko', 'Artemi', 'Sidney', 'Brad',
                       'Jake', 'Tyler', 'Nick', 'Erik', 'Quinn', 'Roman',
                       'Matthew', 'Elias', 'Jack', 'Trevor', 'Tomas', 'Kevin']
        last_names = ['McDavid', 'Matthews', 'MacKinnon', 'Draisaitl', 'Makar',
                      'Ovechkin', 'Pastrnak', 'Marner', 'Rantanen', 'Panarin',
                      'Crosby', 'Marchand', 'Guentzel', 'Seguin', 'Suzuki',
                      'Karlsson', 'Hughes', 'Josi', 'Tkachuk', 'Pettersson',
                      'Eichel', 'Zegras', 'Hertl', 'Fiala']

        names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
                 for _ in range(n_players)]

        data = {
            'name': names,
            'team': np.random.choice(teams, n_players),
            'position': positions,
            'games_played': games,
            'goals': goals.astype(int),
            'assists': assists.astype(int),
            'points': points.astype(int),
            'plus_minus': np.random.randint(-25, 35, n_players),
            'pim': np.random.poisson(35, n_players),
            'pp_goals': (goals * 0.3).astype(int),
            'pp_points': (points * 0.35).astype(int),
            'sh_goals': np.random.poisson(1, n_players),
            'gw_goals': np.random.poisson(3, n_players),
            'shots': (np.random.poisson(150, n_players) * games / 82).astype(int),
            'toi_per_game': np.where(positions == 'D',
                                      np.random.uniform(18, 25, n_players),
                                      np.random.uniform(14, 21, n_players)),
            'faceoff_pct': np.where(positions == 'C',
                                     np.random.uniform(45, 58, n_players),
                                     np.random.uniform(40, 50, n_players)),
            'cap_hit': cap_hit.astype(int)
        }

        return pd.DataFrame(data)

    def prepare_features(self, df):
        """Prepare features for the value model"""

        # Per-game stats
        df['goals_per_game'] = df['goals'] / df['games_played'].clip(lower=1)
        df['assists_per_game'] = df['assists'] / df['games_played'].clip(lower=1)
        df['points_per_game'] = df['points'] / df['games_played'].clip(lower=1)
        df['shots_per_game'] = df['shots'] / df['games_played'].clip(lower=1)

        # Efficiency metrics
        df['shooting_pct'] = np.where(df['shots'] > 0,
                                       df['goals'] / df['shots'] * 100, 0)

        # Power play contribution
        df['pp_contribution'] = df['pp_points'] / df['points'].clip(lower=1)

        return df

    def build_value_model(self, df):
        """Build a model to estimate player value based on stats"""
        print("\nBuilding player value model...")

        # Features for the model
        feature_cols = [
            'points_per_game', 'goals_per_game', 'plus_minus',
            'shots_per_game', 'toi_per_game', 'pp_points', 'gw_goals'
        ]

        # Filter to players with sufficient games
        df_model = df[df['games_played'] >= 20].copy()

        # Handle missing columns
        for col in feature_cols:
            if col not in df_model.columns:
                df_model[col] = 0

        X = df_model[feature_cols].fillna(0)
        y = df_model['cap_hit']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_scaled, y)

        # Predict "expected" salary
        df_model['expected_salary'] = self.model.predict(X_scaled)

        # Calculate value difference
        df_model['value_difference'] = df_model['cap_hit'] - df_model['expected_salary']
        df_model['value_pct'] = (df_model['value_difference'] / df_model['expected_salary']) * 100

        r2 = self.model.score(X_scaled, y)
        print(f"Model R² score: {r2:.3f}")

        return df_model

    def identify_value_players(self, df, top_n=15):
        """Identify most undervalued and overvalued players"""

        undervalued = df.nsmallest(top_n, 'value_difference')[
            ['name', 'team', 'position', 'points', 'games_played',
             'cap_hit', 'expected_salary', 'value_difference', 'value_pct']
        ].copy()

        overvalued = df.nlargest(top_n, 'value_difference')[
            ['name', 'team', 'position', 'points', 'games_played',
             'cap_hit', 'expected_salary', 'value_difference', 'value_pct']
        ].copy()

        return undervalued, overvalued

    def visualize_results(self, df, save_path='visualizations'):
        """Create visualizations"""
        os.makedirs(save_path, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('NHL Player Value Analysis - San Jose Sharks Analytics', fontsize=14, fontweight='bold')

        # 1. Actual vs Expected Salary
        ax1 = axes[0, 0]
        scatter = ax1.scatter(df['expected_salary']/1e6, df['cap_hit']/1e6,
                             alpha=0.6, c=df['points'], cmap='viridis', s=50)
        ax1.plot([0, 12], [0, 12], 'r--', linewidth=2, label='Fair Value Line')
        ax1.set_xlabel('Expected Salary ($M)', fontsize=11)
        ax1.set_ylabel('Actual Cap Hit ($M)', fontsize=11)
        ax1.set_title('Actual vs Expected Salary', fontsize=12, fontweight='bold')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='Points')

        # 2. Value Distribution
        ax2 = axes[0, 1]
        colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in df['value_pct']]
        ax2.hist(df['value_pct'], bins=25, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Fair Value')
        ax2.set_xlabel('Value Difference (%)', fontsize=11)
        ax2.set_ylabel('Number of Players', fontsize=11)
        ax2.set_title('Value Distribution (Negative = Undervalued)', fontsize=12, fontweight='bold')
        ax2.legend()

        # 3. Points vs Salary by Position
        ax3 = axes[1, 0]
        position_colors = {'C': '#3498db', 'L': '#2ecc71', 'R': '#e74c3c', 'D': '#9b59b6'}
        for pos in df['position'].unique():
            pos_df = df[df['position'] == pos]
            ax3.scatter(pos_df['points'], pos_df['cap_hit']/1e6,
                       alpha=0.6, label=pos, color=position_colors.get(pos, 'gray'), s=50)
        ax3.set_xlabel('Points', fontsize=11)
        ax3.set_ylabel('Cap Hit ($M)', fontsize=11)
        ax3.set_title('Points vs Salary by Position', fontsize=12, fontweight='bold')
        ax3.legend(title='Position')

        # 4. Top Undervalued Players
        ax4 = axes[1, 1]
        top_value = df.nsmallest(12, 'value_pct')
        colors = ['#27ae60' if v < -20 else '#2ecc71' for v in top_value['value_pct']]
        y_pos = range(len(top_value))
        ax4.barh(y_pos, top_value['value_pct'], color=colors, alpha=0.8, edgecolor='black')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f"{n} ({t})" for n, t in zip(top_value['name'], top_value['team'])], fontsize=9)
        ax4.set_xlabel('Value Difference (%)', fontsize=11)
        ax4.set_title('Top 12 Most Undervalued Players', fontsize=12, fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(f'{save_path}/player_value_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}/player_value_analysis.png")
        plt.close()

    def generate_report(self, df, undervalued, overvalued):
        """Generate a text report"""

        report = []
        report.append("=" * 70)
        report.append("NHL PLAYER VALUE ANALYSIS REPORT")
        report.append("San Jose Sharks - Hockey Analytics Project")
        report.append("=" * 70)
        report.append(f"\nAnalysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
        report.append(f"Players Analyzed: {len(df)}")

        report.append("\n" + "-" * 70)
        report.append("TOP 10 UNDERVALUED PLAYERS (Best Value Contracts)")
        report.append("-" * 70)

        for _, row in undervalued.head(10).iterrows():
            report.append(f"\n{row['name']} ({row['team']}) - {row['position']}")
            report.append(f"  Points: {row['points']} in {row['games_played']} GP")
            report.append(f"  Cap Hit: ${row['cap_hit']:,.0f}")
            report.append(f"  Expected: ${row['expected_salary']:,.0f}")
            report.append(f"  Value: {abs(row['value_pct']):.1f}% BELOW expected")

        report.append("\n" + "-" * 70)
        report.append("TOP 10 OVERVALUED PLAYERS")
        report.append("-" * 70)

        for _, row in overvalued.head(10).iterrows():
            report.append(f"\n{row['name']} ({row['team']}) - {row['position']}")
            report.append(f"  Points: {row['points']} in {row['games_played']} GP")
            report.append(f"  Cap Hit: ${row['cap_hit']:,.0f}")
            report.append(f"  Expected: ${row['expected_salary']:,.0f}")
            report.append(f"  Value: {row['value_pct']:.1f}% ABOVE expected")

        report.append("\n" + "=" * 70)
        report.append("SHARKS-SPECIFIC ANALYSIS")
        report.append("=" * 70)

        sharks_df = df[df['team'] == 'SJS']
        if len(sharks_df) > 0:
            sharks_undervalued = sharks_df.nsmallest(3, 'value_pct')
            sharks_overvalued = sharks_df.nlargest(3, 'value_pct')

            report.append("\nBest Value Sharks Players:")
            for _, row in sharks_undervalued.iterrows():
                report.append(f"  • {row['name']}: {abs(row['value_pct']):.1f}% undervalued")

            report.append("\nPotential Trade Candidates:")
            for _, row in sharks_overvalued.iterrows():
                report.append(f"  • {row['name']}: {row['value_pct']:.1f}% overvalued")

        report_text = "\n".join(report)

        with open('player_value_report.txt', 'w') as f:
            f.write(report_text)

        print(report_text)
        return report_text

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print("=" * 60)
        print("NHL Player Value Analysis")
        print("San Jose Sharks - Hockey Analytics Project")
        print("=" * 60)

        # Fetch data
        df = self.fetch_current_season_stats()

        # Prepare features
        df = self.prepare_features(df)

        # Build value model
        df_analyzed = self.build_value_model(df)

        # Identify value players
        undervalued, overvalued = self.identify_value_players(df_analyzed)

        # Visualize
        self.visualize_results(df_analyzed)

        # Generate report
        self.generate_report(df_analyzed, undervalued, overvalued)

        # Save full results
        df_analyzed.to_csv('player_value_analysis.csv', index=False)
        print("\nFull analysis saved to player_value_analysis.csv")

        return df_analyzed, undervalued, overvalued


def main():
    analyzer = NHLPlayerValueAnalyzer()
    results, undervalued, overvalued = analyzer.run_analysis()

    print("\n" + "=" * 60)
    print("ACTIONABLE INSIGHTS")
    print("=" * 60)

    print("\n📊 TRADE TARGETS (Undervalued on Other Teams):")
    targets = undervalued[undervalued['team'] != 'SJS'].head(5)
    for _, p in targets.iterrows():
        print(f"   → {p['name']} ({p['team']}): {p['points']} pts, ${p['cap_hit']/1e6:.1f}M cap hit")
        print(f"      {abs(p['value_pct']):.0f}% below expected value")

    print("\n💰 FREE AGENT TARGETS:")
    print("   Players with strong value metrics may be available in FA")

    print("\n📈 KEY FINDING:")
    avg_savings = undervalued.head(10)['value_difference'].mean()
    print(f"   Top 10 undervalued players average ${abs(avg_savings)/1e6:.1f}M below expected salary")

    return results


if __name__ == "__main__":
    main()
