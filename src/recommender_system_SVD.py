import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import KFold
from surprise import KNNBasic,KNNWithMeans
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import train_test_split
from collections import defaultdict
import io
from surprise.model_selection import GridSearchCV
import csv
import os


df = pd.read_csv('../data/review_scores.csv')

# >>> df.nunique()
# id             130397
# listing_id       3122
# reviewer_id    120320
# comments       127855
# score           20048


df.reviewer_id.value_counts()

# 11967216     21
# 3960298      16
# 64336918     16
# 26599167     15
# 28712299     14

df.listing_id.value_counts()
# 77157       602
# 3812348     599
# 5471844     583
# 7639521     532
# 3629071     491

reader = Reader(rating_scale=(0,5))
data = Dataset.load_from_df(df[['listing_id', 'reviewer_id', 'score']], reader)
data.read_ratings
trainset, testset = train_test_split(data, test_size=.25)
algo = SVD(n_factors=10, n_epochs=10)

# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)
#0.4722348599764088
from surprise import SVDpp
algo = SVDpp(n_factors=10,n_epochs=10)
algo.fit(trainset)
predictions_svd = algo.test(testset)
accuracy.rmse(predictions_svd)
# RMSE: 0.4726
# 0.4726227687986559

from collections import defaultdict
 
def get_top3_recommendations(predictions, topN = 3):
     
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
     
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_recs[uid] = user_ratings[:topN]
     
    return top_recs

top3_recommendations = get_top3_recommendations(predictions)

i=0;
for uid, user_ratings in top3_recommendations.items():
    print(uid, [iid for (iid, _) in user_ratings])
    i=i+1;
    if(i==5):
        break;


# 20913466 [31375398, 31375398, 49524061]
# 34127668 [17563848, 96053143, 48359025]
# 1273760 [12335711, 10286188, 19641303]
# 25811044 [5802917, 12683382, 32096881]
# 13361855 [30048961, 68264567, 138127570]