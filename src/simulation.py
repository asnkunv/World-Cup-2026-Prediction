import pandas as pd
import numpy as np
import joblib
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent  # points to repo root

model = joblib.load(BASE_DIR / "model/best_model.pkl")
scaler = joblib.load(BASE_DIR / "model/scaler.pkl")
team_stats_df = pd.read_csv(BASE_DIR / "data/team_stats.csv")
qualified = pd.read_csv(BASE_DIR / "data/Qualified.csv")
features_df = pd.read_csv(BASE_DIR / "data/data_features.csv")

country_mapping = {
    'ir iran': 'iran',
    'korea republic': 'south korea',
    'usa': 'united states',
    'usmnt': 'united states',
    'st.vincent/grenadines': 'st. vincent / grenadines',
    'st vincent and the grenadines' : 'st. vincent / grenadines',
    'congo dr' : 'dr congo',
    'el salvador': 'elsalvador',
    'united arab emirates' : 'uae',
    'turkiye' : 'turkey',
    'czech republic' : 'czechia',
    'curaçao' : 'curacao'
}

qualified['Team'] = qualified['Team'].str.lower()
qualified['Team'] = qualified['Team'].replace(country_mapping)

team_stats_df = team_stats_df.rename(columns = {'Unnamed: 0' : 'Team'})

qualified_with_stats = pd.merge(qualified, team_stats_df, on='Team', how='left')

qualified_with_stats = qualified_with_stats[qualified_with_stats['Team']!= 'curacao']
qualified_with_stats = qualified_with_stats[qualified_with_stats['Team']!= 'new zealand']
qualified_with_stats = qualified_with_stats[qualified_with_stats['Team']!= 'norway']
qualified_with_stats = qualified_with_stats[qualified_with_stats['Team']!= 'paraguay']

qualified_with_stats = qualified_with_stats.reset_index(drop=True)

sorted(qualified_with_stats["Team"].unique())

qualified_with_stats = qualified_with_stats.rename(columns = {'FIFA_Rank_Nov2025' : 'elo'})

# For now, create a simple encoder. (Because it is much faster for a machine to work with numbers)
team_encoder = LabelEncoder()
team_encoder.fit(qualified_with_stats['Team'].values)

# Current year for WC2026
CURRENT_YEAR = 2026


def predict_match(team1, team2, team_stats_df, model, scaler=None):
    """
    Predicts match outcome: team1 (home) vs team2 (away)
    Returns: [P(away_win), P(draw), P(home_win)]
    """

    # Get stats for both teams
    team1_stats = team_stats_df[team_stats_df['Team'] == team1].iloc[0]
    team2_stats = team_stats_df[team_stats_df['Team'] == team2].iloc[0]

    # Create feature dictionary matching x_train exactly
    features = {
        'year': CURRENT_YEAR,
        'home_team_elo': team1_stats['elo'],
        'away_team_elo': team2_stats['elo'],
        'elo_diff': team1_stats['elo'] - team2_stats['elo'],
        'home_qualified': 1,  # All teams are qualified
        'away_qualified': 1,

        # Home team features
        'home_team_total_matches': team1_stats['total_matches'],
        'home_team_total_wins': team1_stats['total_wins'],
        'home_team_total_draws': team1_stats['total_draws'],
        'home_team_total_losses': team1_stats['total_losses'],
        'home_team_win_rate': team1_stats['win_rate'],
        'home_team_draw_rate': team1_stats['draw_rate'],
        'home_team_loss_rate': team1_stats['loss_rate'],
        'home_team_goals_scored': team1_stats['goals_scored'],
        'home_team_goals_conceded': team1_stats['goals_conceded'],
        'home_team_avg_goals_scored': team1_stats['avg_goals_scored'],
        'home_team_avg_goals_conceded': team1_stats['avg_goals_conceded'],
        'home_team_goal_diff_per_match': team1_stats['goal_diff_per_match'],
        'home_team_xg_for': team1_stats['xg_for'],
        'home_team_xg_against': team1_stats['xg_against'],
        'home_team_avg_xg_for': team1_stats['avg_xg_for'],
        'home_team_avg_xg_against': team1_stats['avg_xg_against'],
        'home_team_xg_diff_per_match': team1_stats['xg_diff_per_match'],
        'home_team_xg_overperf': team1_stats['xg_overperf'],
        'home_team_yellow_cards': team1_stats['yellow_cards'],
        'home_team_avg_yellow_cards': team1_stats['avg_yellow_cards'],
        'home_team_clean_sheets': team1_stats['clean_sheets'],
        'home_team_clean_sheet_rate': team1_stats['clean_sheet_rate'],

        # Away team features
        'away_team_total_matches': team2_stats['total_matches'],
        'away_team_total_wins': team2_stats['total_wins'],
        'away_team_total_draws': team2_stats['total_draws'],
        'away_team_total_losses': team2_stats['total_losses'],
        'away_team_win_rate': team2_stats['win_rate'],
        'away_team_draw_rate': team2_stats['draw_rate'],
        'away_team_loss_rate': team2_stats['loss_rate'],
        'away_team_goals_scored': team2_stats['goals_scored'],
        'away_team_goals_conceded': team2_stats['goals_conceded'],
        'away_team_avg_goals_scored': team2_stats['avg_goals_scored'],
        'away_team_avg_goals_conceded': team2_stats['avg_goals_conceded'],
        'away_team_goal_diff_per_match': team2_stats['goal_diff_per_match'],
        'away_team_xg_for': team2_stats['xg_for'],
        'away_team_xg_against': team2_stats['xg_against'],
        'away_team_avg_xg_for': team2_stats['avg_xg_for'],
        'away_team_avg_xg_against': team2_stats['avg_xg_against'],
        'away_team_xg_diff_per_match': team2_stats['xg_diff_per_match'],
        'away_team_xg_overperf': team2_stats['xg_overperf'],
        'away_team_yellow_cards': team2_stats['yellow_cards'],
        'away_team_avg_yellow_cards': team2_stats['avg_yellow_cards'],
        'away_team_clean_sheets': team2_stats['clean_sheets'],
        'away_team_clean_sheet_rate': team2_stats['clean_sheet_rate'],

        # Difference features
        'goal_diff': team1_stats['avg_goals_scored'] - team2_stats['avg_goals_scored'],
        'def_diff': team1_stats['avg_goals_conceded'] - team2_stats['avg_goals_conceded'],
        'xg_for_diff': team1_stats['avg_xg_for'] - team2_stats['avg_xg_for'],
        'xg_against_diff': team1_stats['avg_xg_against'] - team2_stats['avg_xg_against'],
        'win_rate_diff': team1_stats['win_rate'] - team2_stats['win_rate'],

        # Encoded team names
        'home_team_encoded': team_encoder.transform([team1])[0],
        'away_team_encoded': team_encoder.transform([team2])[0],
    }

    # Convert to DataFrame with correct column order
    X = pd.DataFrame([features])

    # Ensure column order matches training
    expected_columns = [
        'year', 'home_team_elo', 'away_team_elo', 'elo_diff', 'home_qualified', 'away_qualified',
        'home_team_total_matches', 'home_team_total_wins', 'home_team_total_draws',
        'home_team_total_losses', 'home_team_win_rate', 'home_team_draw_rate', 'home_team_loss_rate',
        'home_team_goals_scored', 'home_team_goals_conceded', 'home_team_avg_goals_scored',
        'home_team_avg_goals_conceded', 'home_team_goal_diff_per_match', 'home_team_xg_for',
        'home_team_xg_against', 'home_team_avg_xg_for', 'home_team_avg_xg_against',
        'home_team_xg_diff_per_match', 'home_team_xg_overperf', 'home_team_yellow_cards',
        'home_team_avg_yellow_cards', 'home_team_clean_sheets', 'home_team_clean_sheet_rate',
        'away_team_total_matches', 'away_team_total_wins', 'away_team_total_draws',
        'away_team_total_losses', 'away_team_win_rate', 'away_team_draw_rate', 'away_team_loss_rate',
        'away_team_goals_scored', 'away_team_goals_conceded', 'away_team_avg_goals_scored',
        'away_team_avg_goals_conceded', 'away_team_goal_diff_per_match', 'away_team_xg_for',
        'away_team_xg_against', 'away_team_avg_xg_for', 'away_team_avg_xg_against',
        'away_team_xg_diff_per_match', 'away_team_xg_overperf', 'away_team_yellow_cards',
        'away_team_avg_yellow_cards', 'away_team_clean_sheets', 'away_team_clean_sheet_rate',
        'goal_diff', 'def_diff', 'xg_for_diff', 'xg_against_diff', 'win_rate_diff',
        'home_team_encoded', 'away_team_encoded'
    ]

    X = X[expected_columns]

    # Scale if needed
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=X.columns)

    # Get probabilities
    probs = model.predict_proba(X)[0]

    return probs

def simulate_match(team1, team2, probabilities):
    """
    Simulates match result based on probabilities
    Returns: winner ('team1', 'team2', or 'draw')
    """
    # Assuming probabilities order: [Away Win, Draw, Home Win]
    outcome = np.random.choice(['team2', 'draw', 'team1'], p=probabilities)
    return outcome


def simulate_group_stage(groups, team_stats_df, model, scaler=None):
    """
    Simulates entire group stage
    Returns: dict with group standings
    """
    group_results = {}

    for group_name, teams in groups.items():
        # Initialize standings
        standings = {team: {'points': 0, 'gf': 0, 'ga': 0, 'gd': 0, 'wins': 0}
                     for team in teams}

        # Get all matchups (6 matches per group)
        matchups = list(combinations(teams, 2))

        for team1, team2 in matchups:
            # Predict probabilities
            probs = predict_match(team1, team2, team_stats_df, model, scaler)

            # Simulate match
            result = simulate_match(team1, team2, probs)

            # Simulate goals based on result
            if result == 'team1':
                goals1, goals2 = np.random.randint(1, 4), np.random.randint(0, 2)
                standings[team1]['points'] += 3
                standings[team1]['wins'] += 1
            elif result == 'team2':
                goals1, goals2 = np.random.randint(0, 2), np.random.randint(1, 4)
                standings[team2]['points'] += 3
                standings[team2]['wins'] += 1
            else:  # draw
                goals1 = goals2 = np.random.randint(0, 3)
                standings[team1]['points'] += 1
                standings[team2]['points'] += 1

            # Update standings
            standings[team1]['gf'] += goals1
            standings[team1]['ga'] += goals2
            standings[team1]['gd'] = standings[team1]['gf'] - standings[team1]['ga']

            standings[team2]['gf'] += goals2
            standings[team2]['ga'] += goals1
            standings[team2]['gd'] = standings[team2]['gf'] - standings[team2]['ga']

        # Sort by points, then goal difference
        sorted_standings = sorted(standings.items(),
                                  key=lambda x: (x[1]['points'], x[1]['gd'], x[1]['gf']),
                                  reverse=True)

        group_results[group_name] = sorted_standings

    return group_results


def get_knockout_teams(group_results):
    """
    Gets top 2 from each group (22 teams for 11 groups)
    Returns: list of qualified teams
    """
    qualified = []

    # Get top 2 from each group
    for group_name, standings in group_results.items():
        qualified.append(standings[0][0])  # 1st place
        qualified.append(standings[1][0])  # 2nd place

    return qualified


def simulate_single_knockout_match(team1, team2, team_stats_df, model, scaler=None):
    """
    Simulates single knockout match (no draws, go to penalties if needed)
    """
    probs = predict_match(team1, team2, team_stats_df, model, scaler)
    result = simulate_match(team1, team2, probs)

    # If draw, 50/50 penalty shootout
    if result == 'draw':
        result = np.random.choice(['team1', 'team2'])

    return team1 if result == 'team1' else team2


def simulate_knockout_round(teams, team_stats_df, model, scaler=None):
    """Simulates one knockout round"""
    winners = []
    for i in range(0, len(teams), 2):
        team1, team2 = teams[i], teams[i + 1]
        winner = simulate_single_knockout_match(team1, team2, team_stats_df, model, scaler)
        winners.append(winner)
    return winners


def simulate_knockout_stage(teams, team_stats_df, model, scaler=None):
    """
    Simulates knockout stage
    Returns: (champion, runner_up, third_place)
    """
    current_round = teams.copy()

    # Continue knockout rounds until we have 4 teams (semi-finals)
    while len(current_round) > 4:
        current_round = simulate_knockout_round(current_round, team_stats_df, model, scaler)

    # Semi-finals
    semi_winners = simulate_knockout_round(current_round, team_stats_df, model, scaler)
    semi_losers = [t for t in current_round if t not in semi_winners]

    # Third place match
    third_place = simulate_single_knockout_match(semi_losers[0], semi_losers[1],
                                                 team_stats_df, model, scaler)

    # Final
    champion = simulate_single_knockout_match(semi_winners[0], semi_winners[1],
                                              team_stats_df, model, scaler)
    runner_up = [t for t in semi_winners if t != champion][0]

    return champion, runner_up, third_place


def simulate_tournament(groups, team_stats_df, model, scaler=None):
    """
    Simulates entire WC2026
    Returns: (champion, runner_up, third_place)
    """
    # Group stage
    group_results = simulate_group_stage(groups, team_stats_df, model, scaler)

    # Get qualified teams
    knockout_teams = get_knockout_teams(group_results)

    # Knockout stage
    champion, runner_up, third_place = simulate_knockout_stage(knockout_teams, team_stats_df, model, scaler)

    return champion, runner_up, third_place


def run_monte_carlo(groups, team_stats_df, model, n_simulations=1, scaler=None):
    """
    Runs tournament simulation N times
    Returns: probability distributions
    """
    results = {
        'champion': {},
        'runner_up': {},
        'third_place': {}
    }

    print(f"Running {n_simulations} WC2026 simulations...\n")

    for i in range(n_simulations):
        if (i + 1) % 1000 == 0:
            print(f" Completed {i + 1}/{n_simulations}")

        champion, runner_up, third = simulate_tournament(groups, team_stats_df, model, scaler)

        # Count results
        results['champion'][champion] = results['champion'].get(champion, 0) + 1
        results['runner_up'][runner_up] = results['runner_up'].get(runner_up, 0) + 1
        results['third_place'][third] = results['third_place'].get(third, 0) + 1

    # Convert to probabilities
    for category in results:
        for team in results[category]:
            results[category][team] = results[category][team] / n_simulations * 100

    return results


def display_results(results):
    """Display tournament prediction results"""

    print("\n" + "=" * 60)
    print(" FIFA WORLD CUP 2026 PREDICTIONS ")
    print("=" * 60)

    # Top 15 favorites to win
    champion_probs = sorted(results['champion'].items(), key=lambda x: x[1], reverse=True)
    print("\n TOP 15 FAVORITES TO WIN:")
    print("-" * 60)
    for i, (team, prob) in enumerate(champion_probs[:15], 1):
        bar = "█" * int(prob / 2)
        print(f"{i:2d}. {team.title():20s} {prob:5.2f}% {bar}")

    # Runner-up probabilities
    print("\n TOP 10 RUNNER-UP PROBABILITIES:")
    print("-" * 60)
    runner_up_probs = sorted(results['runner_up'].items(), key=lambda x: x[1], reverse=True)
    for i, (team, prob) in enumerate(runner_up_probs[:10], 1):
        print(f"{i:2d}. {team.title():20s} {prob:5.2f}%")

    # Third place probabilities
    print("\n TOP 10 THIRD-PLACE PROBABILITIES:")
    print("-" * 60)
    third_probs = sorted(results['third_place'].items(), key=lambda x: x[1], reverse=True)
    for i, (team, prob) in enumerate(third_probs[:10], 1):
        print(f"{i:2d}. {team.title():20s} {prob:5.2f}%")

    print("\n" + "=" * 60)

