import sqlite3
import pandas as pd

# Read sqlite database with pandas
conn = sqlite3.connect('database.sqlite')
match_df = pd.read_sql_query("SELECT * FROM Match", conn)
attributes_df = pd.read_sql_query("SELECT * FROM Team_Attributes", conn)

# Keep only relevant columns for analysis and convert to csvs
attributes_df = attributes_df[['team_api_id',
                               'buildUpPlaySpeed', 'buildUpPlayPassing', 'chanceCreationPassing',
                               'chanceCreationCrossing',
                               'chanceCreationShooting', 'defencePressure', 'defenceAggression', 'defenceTeamWidth']]
match_df = match_df[
    ['home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal', 'B365H', 'B365D', 'B365A', 'BWH',
     'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH',
     'LBD', 'LBA']]
match_df = match_df.dropna()
match_df.to_csv('match_new.csv',index=False)
attributes_df.to_csv('team_attributes_new.csv', index=False)
