# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 20:42:23 2017
@author: roel
"""
import json
import random
from time import time


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer # Vectorize terms
from sklearn.model_selection import train_test_split # Splitting in train & test set
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics # To measure accuracy
from nltk.stem import PorterStemmer # Preprocess - For stemming

random.seed(19880303)

# http://www.trumptwitterarchive.com/
trumptweets = json.load(open('dataset.json',encoding='utf8'))
df_trump = [i['text'] for i in trumptweets if i['is_retweet'] is False]
del trumptweets

# Random dataset: http://followthehashtag.com/datasets/free-twitter-dataset-usa-200000-free-usa-tweets/
randomtweets = open('random.csv', newline='',encoding='utf8')
next(randomtweets) # Remove header
df_random = [row for row in randomtweets]
del randomtweets
df_random = random.sample(df_random,len(df_trump))

# Randomize order of two lists
# Thanks at https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
df_tweets = df_random + df_trump
df_target = [1] * len(df_random) + [0] * len(df_trump)
del df_trump, df_random

df = list(zip(df_tweets,df_target))
random.shuffle(df)
df_tweets, df_target = zip(*df)
df_tweets_train, df_tweets_test, df_target_train, df_target_test = train_test_split(
        df_tweets, df_target, test_size=0.3, random_state=1)

dTrain = {'data': df_tweets_train, 'target': df_target_train}
dTest = {'data': df_tweets_test, 'target': df_target_test}
del df, df_target, df_tweets, df_tweets_train, df_target_train, df_tweets_test, df_target_test

###########################################################
# Document Preprocessing ##################################
###########################################################

def do_stem(train, test):
    # STEMing
    # Thanks at https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn
    stemmer = PorterStemmer()
    analyzer = CountVectorizer().build_analyzer()
    
    def preprocess_words(doc):
        return (stemmer.stem(w) for w in analyzer(doc))
    
    cv_transform = CountVectorizer(analyzer=preprocess_words, 
                               stop_words = 'english',
                               ngram_range = (1,3),
                               min_df=3)
                                   
    # Count occurances. 
    dfo_train = cv_transform.fit_transform(train['data'])
    dfo_test = cv_transform.transform(test['data'])

    # For NMF we need to get densities, so that it accounts for the fact that docs
    # haven't got the same length. So we divide occurences by total doc word count.
    tf_transform = TfidfTransformer(use_idf=False).fit(dfo_train)
    dff_train = tf_transform.fit_transform(dfo_train)
    dff_test = tf_transform.transform(dfo_test)

    # For NMF there is another refinement: adjusting frequency for unique occurence.
    tfidf_transform = TfidfTransformer(use_idf=True).fit(dfo_train)
    dffi_train = tfidf_transform.fit_transform(dfo_train)
    dffi_test = tfidf_transform.transform(dfo_test)

    return {'data': train['data'], 'target': train['target'], 'dfo': dfo_train, 'dff': dff_train, 'dffi': dffi_train}, {'data': test['data'], 'target': test['target'], 'dfo': dfo_test, 'dff': dff_test, 'dffi': dffi_test}

dTrain, dTest  = do_stem(dTrain, dTest)

def benchmark(model, dTe, dTr):
    # Thanks at http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html
    print('_' * 80)
    print("Training: ")
    print(model)
    t0 = time()
    model.fit(dTr['dffi'], dTr['target'])
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = model.predict(dTe['dffi'])
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(dTe['target'], pred)
    print("accuracy:   %0.3f" % score)
    
    model_desc = str(model).split('(')[0]
    print("confusion matrix:")
    print(metrics.confusion_matrix(dTe['target'], pred))
    return model_desc, score, train_time, test_time


results = []
# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01),dTest,dTrain))
results.append(benchmark(BernoulliNB(alpha=.01),dTest,dTrain))

