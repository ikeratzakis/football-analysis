import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate

# Number of cpu cores to be used (-1 means all)
n_jobs = -1


# Function to get result for each match (home goals - away goals), home win->0 draw->1 away win->2
def match_result(goal_diff):
    if goal_diff > 0:
        return 0
    elif goal_diff == 0:
        return 1
    else:
        return 2


# Thanks to kev from https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
# Function that allows us to flatten an irregular list of lists (each element might be int, or list, or anything)
def flatten(xs):
    res = []

    def loop(ys):
        for i in ys:
            if isinstance(i, list):
                loop(i)
            else:
                res.append(i)

    loop(xs)
    return res


# Load csvs and apply result function to column
match_df = pd.read_csv('match_new.csv')
attributes_df = pd.read_csv('team_attributes_new.csv')
match_df['result'] = match_df['home_team_goal'] - match_df['away_team_goal']
match_df['result'] = match_df['result'].apply(match_result)
# print(match_df['result'])

# Create 4 subsets for each betting company (B365 etc)
result = match_df[['result']].values.tolist()
result = [item for sublist in result for item in sublist]
B365 = match_df[['B365H', 'B365D', 'B365A']].values.tolist()
BW = match_df[['BWH', 'BWD', 'BWA']].values.tolist()
IW = match_df[['IWH', 'IWD', 'IWA']].values.tolist()
LB = match_df[['LBH', 'LBD', 'LBA']].values.tolist()

# Least squares for each subset
SGD = make_pipeline(StandardScaler(), SGDClassifier(loss='squared_loss', max_iter=1000, tol=1e-3, n_jobs=n_jobs))
print('Training least squares classifier for B365 company')
print('Least squares score for B365:', np.mean(cross_validate(SGD, B365, result, cv=10, n_jobs=n_jobs)['test_score']))
print('Training least squares classifier for BW company')
print('Least squares score for BW:', np.mean(cross_validate(SGD, BW, result, cv=10, n_jobs=n_jobs)['test_score']))
print('Training least squares classifier for IW company')
print('Least squares score for BW:', np.mean(cross_validate(SGD, IW, result, cv=10, n_jobs=n_jobs)['test_score']))
print('Training least squares classifier for LB company')
print('Least squares score for BW:', np.mean(cross_validate(SGD, LB, result, cv=10, n_jobs=n_jobs)['test_score']))

# Ridge classification for each subset
print('Training ridge classifier for B365 company')
ridge_B365 = RidgeClassifier()
ridge_B365.fit(B365, result)
print('Ridge classifier score for B365:',
      np.mean(cross_validate(ridge_B365, B365, result, cv=10, n_jobs=n_jobs)['test_score']))
print('Training ridge classifier for BW company')
ridge_BW = RidgeClassifier()
ridge_BW.fit(BW, result)
print('Ridge classifier score for BW:',
      np.mean(cross_validate(ridge_BW, BW, result, cv=10, n_jobs=n_jobs)['test_score']))
print('Training ridge classifier for IW company')
ridge_IW = RidgeClassifier()
ridge_IW.fit(IW, result)
print('Ridge classifier score for IW:',
      np.mean(cross_validate(ridge_IW, IW, result, cv=10, n_jobs=n_jobs)['test_score']))
print('Training ridge classifier for LW company')
ridge_LB = RidgeClassifier()
ridge_LB.fit(LB, result)
print('Ridge classifier score for LB:',
      np.mean(cross_validate(ridge_LB, LB, result, cv=10, n_jobs=n_jobs)['test_score']))

# Now combine all features (betting companies stats + team attributes to create a dataset of 28 columns for prediction)
# Create a dict that maps ids to attributes
attributes = attributes_df.values.tolist()
attributes_dict = {}
for attribute in attributes:
    team_id = attribute[0]
    if team_id not in attributes_dict:
        attributes_dict[team_id] = attribute[1:]
    else:
        for x in attribute[1:]:
            attributes_dict[team_id].append(x)
# Because a single team is associated with multiple attributes, we can take their average to construct a final
# dictionary
for key, value in attributes_dict.items():
    average_list = np.mean([value[i:i + 8] for i in range(0, len(value), 8)], axis=0).tolist()
    average_list = [int(i) for i in average_list]
    attributes_dict[key] = average_list
match = match_df[
    ['home_team_api_id', 'away_team_api_id', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH',
     'LBD', 'LBA']].values.tolist()

# Append to full dataset list according to dictionary. If there are any missing ids, remove the corresponding
# results Clean up memory for low memory machines
del B365
del BW
del IW
del LB
del match_df
del attributes_df
full_dataset = []
new_results = []
for (x, y) in zip(match, result):
    # Check if dictionary contains the corresponding team id and append accordingly
    id_1 = int(x[0])
    id_2 = int(x[1])
    if id_1 in attributes_dict and id_2 in attributes_dict:
        for item in x[2:]:
            full_dataset.append(item)
        for team_1_attribute in attributes_dict[id_1]:
            full_dataset.append(team_1_attribute)
        for team_2_attribute in attributes_dict[id_2]:
            full_dataset.append(team_2_attribute)
        new_results.append(y)

# Flatten and format the dataset correctly in chunks of 28
flat_dataset = flatten(full_dataset)
full_dataset = [flat_dataset[i:i + 28] for i in range(0, len(flat_dataset), 28)]

# MLP classifier
print('Training MLP classifier for full dataset')
MLP = make_pipeline(StandardScaler(), MLPClassifier(max_iter=300, verbose=1))
print('MLP classifier score for the full dataset:',
      np.mean(cross_validate(MLP, full_dataset, new_results, cv=10, n_jobs=n_jobs)['test_score']))
