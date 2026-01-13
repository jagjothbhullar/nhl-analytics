# NHL Player Value Analysis

A data analytics project that identifies undervalued and overvalued NHL players by comparing performance metrics to contract value. Built for the San Jose Sharks hockey operations team.

## Overview

This project uses machine learning to estimate a player's "expected salary" based on their on-ice performance, then compares it to their actual cap hit to identify:

- **Undervalued players** - Performing above their contract value (trade targets)
- **Overvalued players** - Performing below their contract value (trade candidates)

## Key Findings

- The model achieves ~55% R² score predicting salary from performance stats
- Top undervalued players average $1.5M below expected salary
- Clear value gaps exist that can inform roster decisions

## Features

- Fetches live data from NHL Stats API
- Builds regression model using points, TOI, plus/minus, and other metrics
- Generates visualizations comparing actual vs expected salary
- Provides Sharks-specific analysis and actionable recommendations

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn requests
```

## Usage

```bash
python nhl_player_value_analysis.py
```

This will:
1. Fetch current NHL player statistics
2. Build the value prediction model
3. Generate visualizations in `/visualizations`
4. Output a report with top undervalued/overvalued players
5. Save full results to `player_value_analysis.csv`

## Output Files

- `player_value_analysis.csv` - Full analysis data
- `player_value_report.txt` - Summary report
- `visualizations/player_value_analysis.png` - Charts and graphs

## Data Sources

- [NHL Stats API](https://api.nhle.com) - Player statistics
- Salary estimates based on league averages and point production

## Future Enhancements

- Integrate real salary data from PuckPedia or CapFriendly
- Add advanced stats (Corsi, Fenwick, xG) from Natural Stat Trick
- Build age curves to project future value
- Create interactive dashboard with Streamlit

## Author

**Jagjoth Bhullar**
Deputy General Counsel, San Jose Sharks
[Portfolio](https://jagjothbhullar.github.io/personal-blog/)

## License

MIT License
