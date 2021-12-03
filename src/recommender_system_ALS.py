import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.mllib.recommendation import ALS
from pyspark.ml.recommendation import ALS as mlals
from pyspark.ml.evaluation import RegressionEvaluator

import math
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.ml.evaluation import RegressionEvaluator

ratings_df = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
.option('multiLine', 'True') \
        .option('escape', '"') \
        .option("mode", "DROPMALFORMED")\
    .csv('../data/reviews_sentiment.csv')

data = ratings_df.select('reviewer_id','listing_id','polarity')

(trainingData,validationData,testData) = data.randomSplit([0.7,0.15,0.15])

validation_for_predict = validationData.select('reviewer_id','listing_id')
test_for_predict = testData.select('reviewer_id','listing_id')


seed = 5 
iterations = 5
regularization_parameter = 0.2 
ranks = [8, 12, 16]
nonnegative = True
min_error = float('inf')
best_rank = 0
best_iteration = -1

for rank in ranks:
    model = ALS.train(trainingData, rank, iterations=iterations,
                      lambda_=regularization_parameter,seed=seed,nonnegative=True)
    predictions = model.predictAll(validation_for_predict.rdd).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validationData.rdd.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()) # RMSE Error
    print ('For rank %s the RMSE is %s' % (rank, error))
    if error < min_error:
        min_error = error
        best_rank = rank

print ('The best model was trained with rank %s and minimum RMSE %s' % (best_rank,min_error))

predictions_test = model.predictAll(test_for_predict.rdd).map(lambda r: ((r[0], r[1]), r[2]))


def getRecommendations(user,testDf,trainDf,model):
    # get all user and his/her rated listings
    userDf = testDf.filter(testDf.reviewer_id == user)
    
    # filter listings from main set which have not been rated by selected user
    # and pass it to model we sreated above
    mov = trainDf.select('listing_id').subtract(userDf.select('listing_id'))
    
    # Again we need to covert our dataframe into RDD
    pred_rat = model.predictAll(mov.rdd.map(lambda x: (user, x[0]))).collect()
    
    # Get the top recommendations
    recommendations = sorted(pred_rat, key=lambda x: x[2], reverse=True)[:3]
    
    return recommendations


reviewer_id = 143771
derived_rec = getRecommendations(reviewer_id,testData,trainingData,model)

print ("listings recommended for:%d" % reviewer_id)

derived_rec

