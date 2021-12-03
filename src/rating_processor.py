import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+zxx

from pyspark.sql import SparkSession, functions, types
from pyspark.sql.types import *
from pyspark.sql.functions import lower
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt
#from wordcloud import WordCloud 
import pandas as pd
import re
import string
import langid

@functions.udf(returnType=types.StringType())
def detect_lang(x):
    return langid.classify(x)[0]


@functions.udf(returnType=ArrayType(StringType()))
def sent_Tokenizer(x):
    return nltk.sent_tokenize(x)


@functions.udf(returnType=ArrayType(StringType()))
def word_Tokenizer(x):
    splitted = [word for line in x for word in line.split()]
    return splitted


@functions.udf(returnType=ArrayType(StringType()))
def remove_stopwords(x):
    from nltk.corpus import stopwords
    stop_words=set(stopwords.words('english'))
    filtered_sentence = [w for w in x if not w in stop_words]
    return filtered_sentence


@functions.udf(returnType=ArrayType(StringType()))
def remove_punctuations(x):
    list_punct=list(string.punctuation)
    filtered = [''.join(c for c in s if c not in list_punct) for s in x] 
    filtered_space = [s for s in filtered if s] #remove empty space 
    return filtered_space


@functions.udf(returnType=ArrayType(StringType()))
def lemmatization(x):
    lemmatizer = WordNetLemmatizer()
    finalLem = [lemmatizer.lemmatize(s) for s in x]
    return finalLem


@functions.udf(returnType=ArrayType(StringType()))
def joinTokens(x):
    x = " ".join(x)
    return [x]
      


@functions.udf(returnType=types.FloatType())
def get_polarity(x):
    for item in x:
        return TextBlob(item).sentiment.polarity


def main(inputs, output):
    reviews_scheme = types.StructType([
                                types.StructField('listing_id', types.IntegerType(), True),
                                types.StructField('id', types.IntegerType(), True),
                                types.StructField('date', types.DateType(), True),
                                types.StructField('reviewer_id', types.IntegerType(), True),
                                types.StructField('reviewer_name', types.StringType(), True),
                                types.StructField('comments', types.StringType(), True)])

    reviews = spark.read.option('multiLine', 'True') \
        .option('escape', '"') \
        .option("mode", "DROPMALFORMED")\
        .csv(inputs, header=True, schema=reviews_scheme)
    
    # filter out null comments
    comments = reviews.filter(reviews["comments"].isNotNull()).select("id", 'reviewer_id', 'listing_id', "comments")
    # lowercase comments
    comments1 = comments.withColumn('comments', lower(comments['comments']))
    # filter out automated posting
    comments1 = comments1.filter(~comments1.comments.contains('this is an automated posting.'))
    # detect comment language
    comments1 = comments1.withColumn('language',detect_lang(comments1['comments']))
    # keep only english comments
    comments1 = comments1.where(comments1['language']== 'en')


    sent_split = comments1.withColumn('comment1', sent_Tokenizer(comments1['comments']))
    word_split = sent_split.withColumn('comment1', word_Tokenizer(sent_split['comment1']))
    removeStopwords_df = word_split.withColumn('comment1', remove_stopwords(word_split['comment1']))
    remove_punct = removeStopwords_df.withColumn('comment1', remove_punctuations(removeStopwords_df['comment1']))
    lemmatized = remove_punct.withColumn('comment1', lemmatization(remove_punct['comment1']))
    join_token = lemmatized.withColumn('comment1', joinTokens(lemmatized['comment1']))
    sentiment = join_token.withColumn('polarity', get_polarity(join_token['comment1']))

    outfile = sentiment.select('id', 'listing_id','reviewer_id', 'comments', 'polarity')
    # outfile.write.option("header", "true").option('multiLine', 'True') \
    # .option('escape', '"') .csv(output, mode='overwrite',encoding='UTF-8')
    output.toPandas().to_csv()


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    spark = SparkSession.builder.appName('airbnb reviews').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main(inputs, output)