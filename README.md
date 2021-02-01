# football-analysis
Predict football game results using sklearn. The dataset is in .sqlite format from here :https://www.kaggle.com/hugomathien/soccer
# Usage
After downloading and unzipping the dataset from the above link, run the csv_cleanup.py script. Then run the predict.py script to get 10fold cross validation scores using different datasets each time (the 3 features that correspond to each of the 4 betting companies' stats). The final dataset consists of 28 columns that contain each playing team's attributes and all the betting companies' statistics.
