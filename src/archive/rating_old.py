import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('airbnb sentiment analysis').getOrCreate()
assert spark.version >= '2.3' # make sure we have Spark 2.3+
spark.sparkContext.setLogLevel('WARN')

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
#from wordcloud import WordCloud 
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pyspark.sql.types import ArrayType, FloatType, StringType,IntegerType
#nltk.download('vader_lexicon')
# review_schema = types.StructType([
#     types.StructField('listing_id', types.IntegerType()),
#     types.StructField('id', types.IntegerType()),
#     types.StructField('date', types.DateType()),
#     types.StructField('reviwer_id', types.IntegerType()),
#     types.StructField('reviewer_name', types.StringType()),
#     types.StructField('comments', types.StringType()),
# ])

def main(input,output):
    #read csv by escape quote and quote with comma and new line etc.
data = spark.read.option("multiline", "true")\
.option("quote", '"')\
.option("header", "true")\
.option("escape", "\\")\
.option("escape", '"').csv(input)

#data row count:159807

data_nona = data.filter(data['comments'].isNotNull())

#data row count: 159754


#reviews to lower case
data_lower = data_nona.withColumn('comments',functions.lower(data_nona['comments']))


#Tokenizer
@functions.udf(returnType=ArrayType(StringType()))
def sent_TokenizeFunct(x):
    return nltk.sent_tokenize(x)

data_sent_tokenizer = data_lower.withColumn('comments',sent_TokenizeFunct(data_lower['comments']))


@functions.udf(returnType=ArrayType(ArrayType(StringType())))
def word_TokenizeFunct(x):
    output_list = []
    for sent in x:
        sent_list = []
        for word in sent.split(' '):
            sent_list+=[word]
        output_list+=[sent_list]
    return output_list

data_word_tokenizier = data_sent_tokenizer.withColumn('comments',word_TokenizeFunct(data_sent_tokenizer['comments']))

@functions.udf(returnType=ArrayType(StringType()))
def removeStopWordsFunct(x):
    stop_words=list(stopwords.words('english'))
    filteredSentence = [w for w in x if not w in stop_words]
    return filteredSentence

data_stopword = data_word_tokenizier.withColumn('comments',removeStopWordsFunct(data_word_tokenizier['comments']))

@functions.udf(returnType=ArrayType(StringType()))
def removePunctuationsFunct(x):
    list_punct=list(string.punctuation)
    filtered = [''.join(c for c in s if c not in list_punct) for s in x] 
    filtered_space = [s for s in filtered if s] #remove empty space 
    return filtered_space

data_nopunctuation = data_stopword.withColumn('comments',removePunctuationsFunct(data_stopword['comments']))

@functions.udf(returnType=ArrayType(StringType()))
def lemmatizationFunct(x):
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    finalLem = [lemmatizer.lemmatize(s) for s in x]
    return finalLem

data_lemmatization = data_nopunctuation.withColumn('comments',lemmatizationFunct(data_nopunctuation['comments']))

@functions.udf(returnType=ArrayType(StringType()))
def joinTokensFunct(x):
    joinedTokens_list = []
    x = " ".join(x)
    return x

data_joinToken = data_lemmatization.withColumn('comments',joinTokensFunct(data_lemmatization['comments']))

@functions.udf(returnType=ArrayType(StringType()))
def extractPhraseFunct(x):
    stop_words=set(stopwords.words('english'))    
    
    def leaves(tree):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
            yield subtree.leaves()
    
    def get_terms(tree):
        for leaf in leaves(tree):
            term = [w for w,t in leaf if not w in stop_words]
            yield term
    sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
    grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        
    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)
    tokens = nltk.regexp_tokenize(x,sentence_re)
    postoks = nltk.tag.pos_tag(tokens) #Part of speech tagging 
    tree = chunker.parse(postoks) #chunking
    terms = get_terms(tree)
    temp_phrases = []
    for term in terms:
        if len(term):
            temp_phrases.append(' '.join(term))
    
    finalPhrase = [w for w in temp_phrases if w] #remove empty lists
    return finalPhrase

data_extractPhrase = data_joinToken.withColumn('comments',removePunctuationsFunct(data_joinToken['comments']))

@functions.udf(returnType=FloatType())
def sentimentWordsFunct(x):
    analyzer = SentimentIntensityAnalyzer() 
    senti_list_temp = []
    for i in x:
        y = ''.join(i) 
        vs = analyzer.polarity_scores(y)
        senti_list_temp.append([y, vs])
        senti_list_temp = [w for w in senti_list_temp if w]
    sentiment_list  = []
    text_length = len(senti_list_temp)
    total_score = 0
    for j in senti_list_temp:
        first = j[0]
        second = j[1]
        for (k,v) in second.items():
            if k == 'compound':
                total_score+=v   
    return total_score/text_length

data_sentiment = data_extractPhrase.withColumn('comments',sentimentWordsFunct(data_extractPhrase['comments']))
data_sentiment.repartition(1).write.csv('test')

#sentiment analysis



#build pipeline


#save output as pipelinemodel instance
#weather_model.write().overwrite().save(model_file)

if __name__ == '__main__':
    input = '../data/reviews.csv'
    output = '../outputs'
    #model_file = sys.argv[2]
    main(input,output)