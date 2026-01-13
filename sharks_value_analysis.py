"""
San Jose Sharks Player Value Analysis
======================================
Focused analysis of Sharks roster with real salary data,
advanced stats, and age curve projections.

Author: Jagjoth Bhullar
Deputy General Counsel, San Jose Sharks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# REAL SHARKS ROSTER DATA (2024-25 Season)
# Sources: CapWages, Hockey-Reference, PuckPedia
# =============================================================================

SHARKS_ROSTER = [
    # Forwards
    {"name": "Macklin Celebrini", "position": "C", "age": 19, "cap_hit": 975000,
     "gp": 70, "goals": 25, "assists": 38, "points": 63, "plus_minus": -31,
     "toi_per_game": 19.8, "shots": 236, "contract_years": 3, "status": "ELC"},

    {"name": "William Eklund", "position": "LW", "age": 23, "cap_hit": 863000,
     "gp": 77, "goals": 17, "assists": 41, "points": 58, "plus_minus": -7,
     "toi_per_game": 19.6, "shots": 151, "contract_years": 1, "status": "ELC",
     "extension_aav": 5600000},

    {"name": "Tyler Toffoli", "position": "RW", "age": 33, "cap_hit": 6000000,
     "gp": 78, "goals": 30, "assists": 24, "points": 54, "plus_minus": -21,
     "toi_per_game": 16.9, "shots": 233, "contract_years": 3, "status": "Standard"},

    {"name": "Will Smith", "position": "C", "age": 20, "cap_hit": 950000,
     "gp": 74, "goals": 18, "assists": 27, "points": 45, "plus_minus": -15,
     "toi_per_game": 15.2, "shots": 142, "contract_years": 3, "status": "ELC"},

    {"name": "Mikael Granlund", "position": "C", "age": 33, "cap_hit": 5000000,
     "gp": 52, "goals": 15, "assists": 30, "points": 45, "plus_minus": -5,
     "toi_per_game": 18.5, "shots": 98, "contract_years": 0, "status": "UFA"},

    {"name": "Fabian Zetterlund", "position": "RW", "age": 26, "cap_hit": 1450000,
     "gp": 64, "goals": 17, "assists": 19, "points": 36, "plus_minus": 8,
     "toi_per_game": 14.8, "shots": 135, "contract_years": 0, "status": "RFA"},

    {"name": "Alexander Wennberg", "position": "C", "age": 31, "cap_hit": 5000000,
     "gp": 65, "goals": 8, "assists": 22, "points": 30, "plus_minus": -18,
     "toi_per_game": 16.2, "shots": 78, "contract_years": 2, "status": "Standard"},

    {"name": "Logan Couture", "position": "C", "age": 36, "cap_hit": 8000000,
     "gp": 25, "goals": 3, "assists": 8, "points": 11, "plus_minus": -12,
     "toi_per_game": 15.5, "shots": 42, "contract_years": 0, "status": "UFA-Injured"},

    {"name": "Barclay Goodrow", "position": "LW", "age": 32, "cap_hit": 3642000,
     "gp": 60, "goals": 5, "assists": 10, "points": 15, "plus_minus": -15,
     "toi_per_game": 13.2, "shots": 65, "contract_years": 0, "status": "UFA"},

    {"name": "Jeff Skinner", "position": "LW", "age": 33, "cap_hit": 3000000,
     "gp": 55, "goals": 12, "assists": 8, "points": 20, "plus_minus": -10,
     "toi_per_game": 13.5, "shots": 110, "contract_years": 1, "status": "Standard"},

    {"name": "Ty Dellandrea", "position": "C", "age": 25, "cap_hit": 1300000,
     "gp": 45, "goals": 6, "assists": 8, "points": 14, "plus_minus": -8,
     "toi_per_game": 12.1, "shots": 55, "contract_years": 1, "status": "Standard"},

    {"name": "Adam Gaudette", "position": "C", "age": 29, "cap_hit": 2000000,
     "gp": 50, "goals": 8, "assists": 6, "points": 14, "plus_minus": -5,
     "toi_per_game": 11.5, "shots": 68, "contract_years": 1, "status": "Standard"},

    {"name": "Ryan Reaves", "position": "RW", "age": 38, "cap_hit": 1350000,
     "gp": 55, "goals": 2, "assists": 4, "points": 6, "plus_minus": -8,
     "toi_per_game": 8.5, "shots": 35, "contract_years": 1, "status": "Standard"},

    # Defensemen
    {"name": "Mario Ferraro", "position": "D", "age": 27, "cap_hit": 3250000,
     "gp": 75, "goals": 5, "assists": 18, "points": 23, "plus_minus": -12,
     "toi_per_game": 21.5, "shots": 95, "contract_years": 3, "status": "Standard"},

    {"name": "Dmitry Orlov", "position": "D", "age": 34, "cap_hit": 6500000,
     "gp": 70, "goals": 4, "assists": 20, "points": 24, "plus_minus": -25,
     "toi_per_game": 22.8, "shots": 88, "contract_years": 1, "status": "Standard"},

    {"name": "Timothy Liljegren", "position": "D", "age": 26, "cap_hit": 3000000,
     "gp": 55, "goals": 3, "assists": 12, "points": 15, "plus_minus": -10,
     "toi_per_game": 18.2, "shots": 62, "contract_years": 1, "status": "Standard"},

    {"name": "Nick Leddy", "position": "D", "age": 34, "cap_hit": 4000000,
     "gp": 65, "goals": 2, "assists": 14, "points": 16, "plus_minus": -18,
     "toi_per_game": 19.5, "shots": 58, "contract_years": 1, "status": "Standard"},

    {"name": "Sam Dickinson", "position": "D", "age": 19, "cap_hit": 942000,
     "gp": 40, "goals": 4, "assists": 10, "points": 14, "plus_minus": -5,
     "toi_per_game": 16.8, "shots": 48, "contract_years": 3, "status": "ELC"},

    {"name": "John Klingberg", "position": "D", "age": 33, "cap_hit": 4000000,
     "gp": 20, "goals": 1, "assists": 5, "points": 6, "plus_minus": -8,
     "toi_per_game": 17.5, "shots": 25, "contract_years": 1, "status": "Injured"},

    # Goalies
    {"name": "Alex Nedeljkovic", "position": "G", "age": 30, "cap_hit": 2500000,
     "gp": 45, "goals": 0, "assists": 1, "points": 1, "plus_minus": 0,
     "toi_per_game": 55.0, "shots": 0, "contract_years": 1, "status": "Standard",
     "sv_pct": 0.895, "gaa": 3.42},

    {"name": "Yaroslav Askarov", "position": "G", "age": 23, "cap_hit": 2000000,
     "gp": 35, "goals": 0, "assists": 2, "points": 2, "plus_minus": 0,
     "toi_per_game": 55.0, "shots": 0, "contract_years": 2, "status": "Standard",
     "sv_pct": 0.908, "gaa": 2.95},
]


class SharksValueAnalyzer:
    """Comprehensive Sharks roster value analysis with age curves"""

    def __init__(self):
        self.df = pd.DataFrame(SHARKS_ROSTER)
        self.age_curves = {}

    def calculate_per_game_stats(self):
        """Calculate per-game metrics"""
        self.df['ppg'] = self.df['points'] / self.df['gp'].clip(lower=1)
        self.df['gpg'] = self.df['goals'] / self.df['gp'].clip(lower=1)
        self.df['apg'] = self.df['assists'] / self.df['gp'].clip(lower=1)
        self.df['shots_per_game'] = self.df['shots'] / self.df['gp'].clip(lower=1)
        self.df['shooting_pct'] = np.where(
            self.df['shots'] > 0,
            self.df['goals'] / self.df['shots'] * 100,
            0
        )

    def build_age_curves(self):
        """
        Build age curves based on NHL research

        Research shows:
        - Forwards peak at age 24-25
        - Defensemen peak at age 25-27
        - Decline accelerates after 30
        - Goalies have longer primes (25-32)
        """
        print("\nBuilding age curves based on NHL research...")

        # Forward age curve (peak at 24-25)
        def forward_curve(age):
            peak_age = 24.5
            if age < peak_age:
                # Growth phase: rapid improvement
                return 0.70 + 0.30 * (age - 18) / (peak_age - 18)
            else:
                # Decline phase: gradual then accelerating
                years_past_peak = age - peak_age
                if years_past_peak < 5:
                    return 1.0 - 0.02 * years_past_peak
                else:
                    return 0.90 - 0.04 * (years_past_peak - 5)

        # Defenseman age curve (peak at 26)
        def defense_curve(age):
            peak_age = 26
            if age < peak_age:
                return 0.65 + 0.35 * (age - 18) / (peak_age - 18)
            else:
                years_past_peak = age - peak_age
                if years_past_peak < 4:
                    return 1.0 - 0.015 * years_past_peak
                else:
                    return 0.94 - 0.035 * (years_past_peak - 4)

        # Goalie age curve (peak at 27-28)
        def goalie_curve(age):
            peak_age = 27.5
            if age < peak_age:
                return 0.75 + 0.25 * (age - 20) / (peak_age - 20)
            else:
                years_past_peak = age - peak_age
                return max(0.5, 1.0 - 0.025 * years_past_peak)

        self.age_curves = {
            'F': forward_curve,
            'D': defense_curve,
            'G': goalie_curve
        }

        # Apply age curve factors
        def get_position_type(pos):
            if pos == 'G':
                return 'G'
            elif pos == 'D':
                return 'D'
            else:
                return 'F'

        self.df['position_type'] = self.df['position'].apply(get_position_type)
        self.df['current_age_factor'] = self.df.apply(
            lambda row: self.age_curves[row['position_type']](row['age']), axis=1
        )

        # Project future age factors (1, 2, 3 years out)
        for years in [1, 2, 3]:
            self.df[f'age_factor_{years}yr'] = self.df.apply(
                lambda row: self.age_curves[row['position_type']](row['age'] + years), axis=1
            )

    def calculate_expected_value(self):
        """
        Calculate expected salary based on production

        Uses market rates:
        - ~$100K per point for forwards (adjusted by age curve)
        - Defensemen valued differently (more TOI, less points)
        - Goalies based on save percentage and workload
        """
        print("Calculating expected player values...")

        # Base value per point (market rate)
        point_value = 110000  # ~$110K per point

        expected_values = []

        for _, player in self.df.iterrows():
            if player['position'] == 'G':
                # Goalie valuation: based on sv%, games, age
                sv_pct = player.get('sv_pct', 0.900)
                base = 1500000
                sv_bonus = (sv_pct - 0.900) * 50000000  # $500K per .001 sv%
                games_factor = min(player['gp'] / 50, 1.2)
                expected = (base + sv_bonus) * games_factor * player['current_age_factor']
            else:
                # Skater valuation
                # Points-based value
                points_82 = player['ppg'] * 82  # Extrapolate to full season
                base_value = points_82 * point_value

                # Adjust for position (D get premium for non-scoring contributions)
                if player['position'] == 'D':
                    base_value *= 1.15
                    # TOI premium for top-pair minutes
                    if player['toi_per_game'] > 20:
                        base_value *= 1.1

                # Age curve adjustment
                expected = base_value * player['current_age_factor']

                # Minimum value (league minimum ~$775K)
                expected = max(expected, 775000)

            expected_values.append(expected)

        self.df['expected_value'] = expected_values
        self.df['value_difference'] = self.df['cap_hit'] - self.df['expected_value']
        self.df['value_pct'] = (self.df['value_difference'] / self.df['expected_value']) * 100

    def project_future_value(self):
        """Project player value over next 3 years"""
        print("Projecting future values...")

        for years in [1, 2, 3]:
            future_age_factor = self.df[f'age_factor_{years}yr']
            current_age_factor = self.df['current_age_factor']

            # Adjust expected value by future age curve
            ratio = future_age_factor / current_age_factor.clip(lower=0.5)
            self.df[f'projected_value_{years}yr'] = self.df['expected_value'] * ratio

            # For players on multi-year deals, compare to cap hit
            self.df[f'future_value_diff_{years}yr'] = np.where(
                self.df['contract_years'] > years,
                self.df['cap_hit'] - self.df[f'projected_value_{years}yr'],
                np.nan  # Contract expires
            )

    def identify_value_players(self):
        """Categorize players by value"""

        # Undervalued: producing above cap hit
        undervalued = self.df[self.df['value_pct'] < -15].sort_values('value_pct')

        # Overvalued: producing below cap hit
        overvalued = self.df[self.df['value_pct'] > 15].sort_values('value_pct', ascending=False)

        # Fair value
        fair_value = self.df[(self.df['value_pct'] >= -15) & (self.df['value_pct'] <= 15)]

        return undervalued, overvalued, fair_value

    def visualize_results(self, save_path='visualizations'):
        """Create comprehensive visualizations"""
        os.makedirs(save_path, exist_ok=True)

        fig = plt.figure(figsize=(16, 14))
        fig.suptitle('San Jose Sharks Player Value Analysis\n2024-25 Season',
                     fontsize=16, fontweight='bold', y=1.02)

        # 1. Value Distribution (scatter plot)
        ax1 = fig.add_subplot(2, 2, 1)
        colors = ['#00a854' if v < -10 else '#ff4444' if v > 10 else '#888888'
                  for v in self.df['value_pct']]
        scatter = ax1.scatter(
            self.df['expected_value'] / 1e6,
            self.df['cap_hit'] / 1e6,
            c=colors, s=100, alpha=0.7, edgecolors='black'
        )

        # Add player labels for notable players
        for _, row in self.df.iterrows():
            if abs(row['value_pct']) > 20 or row['cap_hit'] > 5000000:
                ax1.annotate(
                    row['name'].split()[-1],  # Last name only
                    (row['expected_value']/1e6, row['cap_hit']/1e6),
                    fontsize=8, alpha=0.8
                )

        max_val = max(self.df['cap_hit'].max(), self.df['expected_value'].max()) / 1e6 + 1
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Fair Value')
        ax1.set_xlabel('Expected Value ($M)', fontsize=11)
        ax1.set_ylabel('Actual Cap Hit ($M)', fontsize=11)
        ax1.set_title('Cap Hit vs Expected Value', fontsize=12, fontweight='bold')
        ax1.legend()

        # 2. Age Curves Visualization
        ax2 = fig.add_subplot(2, 2, 2)
        ages = np.arange(18, 40, 0.5)

        for pos_type, curve_func in self.age_curves.items():
            values = [curve_func(a) for a in ages]
            label = {'F': 'Forwards', 'D': 'Defensemen', 'G': 'Goalies'}[pos_type]
            ax2.plot(ages, values, linewidth=2, label=label)

        # Mark Sharks players on the curves
        for _, player in self.df.iterrows():
            ax2.scatter(
                player['age'],
                player['current_age_factor'],
                s=80, zorder=5, alpha=0.7
            )

        ax2.set_xlabel('Age', fontsize=11)
        ax2.set_ylabel('Performance Factor', fontsize=11)
        ax2.set_title('Age Curves with Sharks Players', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.set_ylim(0.4, 1.1)
        ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)

        # 3. Value by Player (horizontal bar chart)
        ax3 = fig.add_subplot(2, 2, 3)

        # Sort by value percentage
        sorted_df = self.df.sort_values('value_pct')
        colors = ['#00a854' if v < -10 else '#ff4444' if v > 10 else '#888888'
                  for v in sorted_df['value_pct']]

        y_pos = range(len(sorted_df))
        ax3.barh(y_pos, sorted_df['value_pct'], color=colors, alpha=0.8, edgecolor='black')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(sorted_df['name'], fontsize=8)
        ax3.axvline(x=0, color='black', linewidth=1)
        ax3.axvline(x=-15, color='green', linestyle='--', alpha=0.5)
        ax3.axvline(x=15, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Value Difference (%)', fontsize=11)
        ax3.set_title('Player Value Rankings\n(Green = Undervalued, Red = Overvalued)',
                      fontsize=12, fontweight='bold')

        # 4. Future Value Projections
        ax4 = fig.add_subplot(2, 2, 4)

        # Focus on key players
        key_players = ['Macklin Celebrini', 'William Eklund', 'Will Smith',
                       'Tyler Toffoli', 'Mario Ferraro', 'Yaroslav Askarov']
        key_df = self.df[self.df['name'].isin(key_players)].copy()

        x = np.arange(len(key_players))
        width = 0.2

        current_vals = key_df['expected_value'] / 1e6
        yr1_vals = key_df['projected_value_1yr'] / 1e6
        yr2_vals = key_df['projected_value_2yr'] / 1e6
        yr3_vals = key_df['projected_value_3yr'] / 1e6

        ax4.bar(x - 1.5*width, current_vals, width, label='Current', color='#3498db')
        ax4.bar(x - 0.5*width, yr1_vals, width, label='+1 Year', color='#2ecc71')
        ax4.bar(x + 0.5*width, yr2_vals, width, label='+2 Years', color='#f1c40f')
        ax4.bar(x + 1.5*width, yr3_vals, width, label='+3 Years', color='#e74c3c')

        ax4.set_xlabel('Player', fontsize=11)
        ax4.set_ylabel('Expected Value ($M)', fontsize=11)
        ax4.set_title('Projected Value by Year (Key Players)', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([n.split()[-1] for n in key_players], rotation=45, ha='right')
        ax4.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(f'{save_path}/sharks_value_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {save_path}/sharks_value_analysis.png")
        plt.close()

    def generate_report(self):
        """Generate comprehensive text report"""

        undervalued, overvalued, fair_value = self.identify_value_players()

        report = []
        report.append("=" * 75)
        report.append("SAN JOSE SHARKS PLAYER VALUE ANALYSIS")
        report.append("2024-25 Season - Confidential Internal Report")
        report.append("=" * 75)
        report.append(f"\nAnalysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
        report.append(f"Players Analyzed: {len(self.df)}")
        report.append(f"Total Cap Commitment: ${self.df['cap_hit'].sum():,.0f}")

        # Executive Summary
        report.append("\n" + "=" * 75)
        report.append("EXECUTIVE SUMMARY")
        report.append("=" * 75)

        total_under = self.df[self.df['value_pct'] < -15]['value_difference'].sum()
        total_over = self.df[self.df['value_pct'] > 15]['value_difference'].sum()

        report.append(f"\n• Undervalued contracts saving: ${abs(total_under):,.0f}")
        report.append(f"• Overvalued contracts costing: ${total_over:,.0f}")
        report.append(f"• Net roster value efficiency: ${abs(total_under) - total_over:,.0f}")

        # Top Value Contracts
        report.append("\n" + "-" * 75)
        report.append("TOP VALUE CONTRACTS (Undervalued - Outperforming Cap Hit)")
        report.append("-" * 75)

        for _, row in undervalued.head(6).iterrows():
            report.append(f"\n{row['name']} ({row['position']}, Age {row['age']})")
            report.append(f"  Cap Hit: ${row['cap_hit']:>10,.0f}")
            report.append(f"  Expected Value: ${row['expected_value']:>10,.0f}")
            report.append(f"  Value: {abs(row['value_pct']):>5.1f}% UNDERVALUED")
            report.append(f"  Production: {row['points']} pts in {row['gp']} GP ({row['ppg']:.2f} PPG)")

            # Future projection
            if row['contract_years'] > 1:
                future_val = row['projected_value_2yr']
                report.append(f"  2-Year Projection: ${future_val:,.0f} expected value")

        # Overvalued Contracts
        report.append("\n" + "-" * 75)
        report.append("CONCERNING CONTRACTS (Overvalued - Underperforming Cap Hit)")
        report.append("-" * 75)

        for _, row in overvalued.head(5).iterrows():
            report.append(f"\n{row['name']} ({row['position']}, Age {row['age']})")
            report.append(f"  Cap Hit: ${row['cap_hit']:>10,.0f}")
            report.append(f"  Expected Value: ${row['expected_value']:>10,.0f}")
            report.append(f"  Value: {row['value_pct']:>5.1f}% OVERVALUED")
            report.append(f"  Contract Status: {row['status']}, {row['contract_years']} years remaining")

        # Age Curve Analysis
        report.append("\n" + "=" * 75)
        report.append("AGE CURVE ANALYSIS - FUTURE PROJECTIONS")
        report.append("=" * 75)

        report.append("\n📈 ASCENDING PLAYERS (Pre-Peak):")
        ascending = self.df[(self.df['age'] < 25) & (self.df['position'] != 'G')]
        for _, row in ascending.iterrows():
            years_to_peak = 25 - row['age']
            peak_factor = self.age_curves[row['position_type']](25)
            current_factor = row['current_age_factor']
            improvement = (peak_factor / current_factor - 1) * 100
            report.append(f"  • {row['name']} (Age {row['age']}): ~{improvement:.0f}% improvement expected by peak")

        report.append("\n📉 DECLINING PLAYERS (Post-Peak):")
        declining = self.df[(self.df['age'] > 30) & (self.df['position'] != 'G')]
        for _, row in declining.sort_values('age', ascending=False).head(5).iterrows():
            decline_3yr = (1 - row['age_factor_3yr'] / row['current_age_factor']) * 100
            report.append(f"  • {row['name']} (Age {row['age']}): ~{decline_3yr:.0f}% decline projected over 3 years")

        # Contract Recommendations
        report.append("\n" + "=" * 75)
        report.append("CONTRACT RECOMMENDATIONS")
        report.append("=" * 75)

        # Extension priorities
        report.append("\n🔒 EXTENSION PRIORITIES:")
        extension_candidates = self.df[
            (self.df['value_pct'] < -10) &
            (self.df['contract_years'] <= 1) &
            (self.df['age'] < 28)
        ]
        for _, row in extension_candidates.iterrows():
            suggested_aav = row['expected_value'] * 1.15  # 15% premium for extension
            report.append(f"  • {row['name']}: Suggest ${suggested_aav/1e6:.1f}M AAV extension")

        # Trade candidates
        report.append("\n🔄 POTENTIAL TRADE CANDIDATES:")
        trade_candidates = self.df[
            (self.df['value_pct'] > 20) |
            ((self.df['age'] > 32) & (self.df['cap_hit'] > 3000000))
        ]
        for _, row in trade_candidates.iterrows():
            report.append(f"  • {row['name']}: ${row['cap_hit']/1e6:.1f}M cap hit, {row['value_pct']:.0f}% overvalued")

        # Core players
        report.append("\n⭐ CORE BUILDING BLOCKS:")
        core = self.df[
            (self.df['age'] < 26) &
            (self.df['ppg'] > 0.5)
        ].sort_values('projected_value_3yr', ascending=False)
        for _, row in core.head(4).iterrows():
            report.append(f"  • {row['name']}: Projected ${row['projected_value_3yr']/1e6:.1f}M value in 3 years")

        report_text = "\n".join(report)

        with open('sharks_value_report.txt', 'w') as f:
            f.write(report_text)

        print(report_text)
        return report_text

    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("=" * 60)
        print("San Jose Sharks Player Value Analysis")
        print("With Real Salary Data and Age Curve Projections")
        print("=" * 60)

        # Calculate stats
        self.calculate_per_game_stats()

        # Build age curves
        self.build_age_curves()

        # Calculate values
        self.calculate_expected_value()

        # Project future
        self.project_future_value()

        # Visualize
        self.visualize_results()

        # Generate report
        self.generate_report()

        # Save data
        self.df.to_csv('sharks_player_values.csv', index=False)
        print("\nData saved to sharks_player_values.csv")

        return self.df


def main():
    analyzer = SharksValueAnalyzer()
    results = analyzer.run_analysis()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS FOR HOCKEY OPERATIONS")
    print("=" * 60)

    # Best value contracts
    best_value = results.nsmallest(3, 'value_pct')
    print("\n💰 BEST VALUE CONTRACTS:")
    for _, p in best_value.iterrows():
        print(f"   {p['name']}: ${p['cap_hit']/1e6:.2f}M for ${p['expected_value']/1e6:.2f}M production")

    # Rising stars
    rising = results[(results['age'] < 24) & (results['ppg'] > 0.6)]
    print("\n🚀 RISING STARS (Lock Up Long-Term):")
    for _, p in rising.iterrows():
        print(f"   {p['name']} (Age {p['age']}): {p['ppg']:.2f} PPG, only ${p['cap_hit']/1e6:.2f}M")

    # Cap concerns
    concerns = results[results['value_pct'] > 25]
    print("\n⚠️  CAP CONCERNS:")
    for _, p in concerns.iterrows():
        print(f"   {p['name']}: ${p['cap_hit']/1e6:.1f}M cap hit, {p['value_pct']:.0f}% above expected")

    return results


if __name__ == "__main__":
    main()
