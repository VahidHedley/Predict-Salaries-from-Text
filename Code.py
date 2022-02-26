# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:32:59 2022

@author: Vahid Hedley
"""
#*****************************************************************************
#                             FOR THE GRADER 

#              Lines 15:199 --> Code for the main model 
#              Lines 200:299 --> Code for exploratory models
#              Lines 300:345 --> Answer to Question 2

#*****************************************************************************
# import packages
import pandas as pd
import numpy as np
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')

# upload dataset
df = pd.read_csv('C:/Users/Vahid Hedley/OneDrive/McGill - Winter 2022/INSY 699 - Text Analytics/Assignments/Individual Project/Train_rev1.csv')

# generate a subset
df = df.sample(n=2500)
df.reset_index(drop=True, inplace=True)

# encode the observations for below and above the salary threshold
df['salary_bucket'] = np.where(df['SalaryNormalized'] <= np.percentile(df['SalaryNormalized'], 75), 'Low', 'High')

# drop all other attributes not pertaining to the object of analysis 
df.drop(df.columns.difference(['FullDescription','salary_bucket']), 1, inplace=True)

# initialize empty list for data pre-processing
df = df.copy()
df['description_tokens'] = ''
df['description_tokens_POS'] = ''
df['description_lemma'] = ''
df['description_lemma_afterstopword'] = ''
df['clean_character'] = ''
df['clean_numerical'] = ''
df['clean_smallw'] = ''
df['clean_english'] = ''
df['clean_duplicate'] = ''
df['final_message'] = ''
w = WordNetLemmatizer()

# function for part of speech tagging
def get_wordnet_pos(lst):
    l = []
    for word in lst:
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        l.append(tag_dict.get(tag, wordnet.NOUN))
    return l

# function for lemmatization
def get_lemma(l1, l2):
    l = []
    for i in range(len(l1)):
        itm=w.lemmatize(l1[i], pos=l2[i])
        l.append(itm.lower())
    return l

# function for removing duplicate characters
def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist

# function for merging list 
def convert(lst):
    return ' '.join(lst)

# tokenize the text field
for i in range(len(df)):
    df['description_tokens'][i] = word_tokenize(df.loc[i,'FullDescription'])
    
# part of speech tagging
for i in range(len(df)):
    df['description_tokens_POS'][i] = get_wordnet_pos(df.loc[i,'description_tokens'])

# lemmatization
for i in range(len(df)):
    df['description_lemma'][i] = get_lemma(df['description_tokens'][i], df['description_tokens_POS'][i])

# remove stopwords 
df['description_lemma_afterstopword'] = [list() for x in range(len(df.index))]
stop_words = set(stopwords.words('english'))
for i in range(len(df)):
    for j in df['description_lemma'][i]:
        if j not in stop_words:
            df['description_lemma_afterstopword'][i].append(j)
    
# remove punctuation 
for i in range(len(df)):
    df['clean_character'][i] = [word for word in df['description_lemma_afterstopword'][i] if word.isalnum()]

# remove numerical characters
for i in range(len(df)):
    df['clean_numerical'][i] = [word for word in df['clean_character'][i] if not word.isdigit()]

# remove words with less than three characters 
for i in range(len(df)):
    df['clean_smallw'][i] = [word for word in df['clean_numerical'][i] if len(word) >= 4]

# remove non-english words 
df['clean_english'] = [list() for x in range(len(df.index))]
words = set(nltk.corpus.words.words())
for i in range(len(df)):
    for j in df['clean_smallw'][i]:
        if j in words:
            df['clean_english'][i].append(j)
            
# remove duplicate words 
for i in range(len(df)):
    df['clean_duplicate'][i] = unique_list(df.loc[i,'clean_english'])

# convert back into a list 
for i in range(len(df)):
    df['final_message'][i] = convert(df.loc[i,'clean_duplicate'])

# drop unnecessary features
df.drop(df.columns.difference(['final_message','salary_bucket']), 1, inplace=True)

# export pre-processed dataset
df.to_csv('C:/Users/0/OneDrive/McGill - Winter 2022/INSY 699 - Text Analytics/Assignments/Individual Project/FINAL_Preprocessed.csv', index=False)

# prepare dataset for model fitting
df2 = df.copy()
high = df2.loc[df2['salary_bucket'] == 'High']  # high salary
low = df2.loc[df2['salary_bucket'] == 'Low']    # low salary
high.drop('salary_bucket', axis=1, inplace=True)
low.drop('salary_bucket', axis=1, inplace=True)
high.columns = ["text"]
low.columns = ["text"]
high.reset_index(drop=True, inplace=True)
low.reset_index(drop=True, inplace=True)

# restructure dataframe and convert to tuple 
data = ([(h['text'], 'High') for index, h in high.iterrows()]+
    [(l['text'], 'Low') for index, l in low.iterrows()])

# re-tokenize the dataset
tokens = set(word.lower() for words in data for word in word_tokenize(words[0]))
train = [({word: (word in word_tokenize(x[0])) \
            for word in tokens}, x[1]) for x in data] 

# shuffle the training data
random.shuffle(train)

# split into training (80%) and test (20%) sets
train_x = train[:500]
test_x = train[500:]

# fit the model using NLTK Naive Bayes Classifier
model = nltk.NaiveBayesClassifier.train(train_x)

# the top 10 most important features for each class label
model.show_most_informative_features(10)

#-----------------------------------------------------------------------------
# Most Informative Features
#                   depth = True             High : Low    =     14.7 : 1.0
#               abundance = True             High : Low    =     10.2 : 1.0
#              consulting = True             High : Low    =     10.2 : 1.0
#                  entity = True             High : Low    =     10.2 : 1.0
#                  master = True             High : Low    =     10.2 : 1.0
#                  spread = True             High : Low    =     10.2 : 1.0
#                  vendor = True             High : Low    =     10.2 : 1.0
#                offshore = True             High : Low    =      8.8 : 1.0
#                  agenda = True             High : Low    =      7.9 : 1.0
#                   arise = True             High : Low    =      7.9 : 1.0
#-----------------------------------------------------------------------------
# interpretation: the ratio of occurences in High and Low income quartiles for   
# every word - ie: the term 'depth' appears 14.7 times more as often among job 
# descriptions for people in the top 75th percentile of salaries 

# check the model prediction accuracy with the test dataset 
a = nltk.classify.accuracy(model, test_x)
print("Accuracy score:", a)

#-----------------------------------------------------------------------------
# Accuracy score: 0.737
#-----------------------------------------------------------------------------

#*****************************************************************************
#*****************************************************************************
#                          END OF MAIN MODEL OUTPUT
#*****************************************************************************
#*****************************************************************************

# here I will explore the several sklearn naive bayes models to see if there is 
# a notable difference in accuracy across algorithms

# import packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import random
import itertools
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# upload pre-processed dataset 
df = pd.read_csv('C:/Users/Vahid Hedley/OneDrive/McGill - Winter 2022/INSY 699 - Text Analytics/Assignments/Individual Project/FINAL_Preprocessed.csv')

# define class names
class_names = ['Low', 'High']

# split into test (80%) and training (20%) sets
df_train = df.sample(n=2000)
df_test = df.sample(n=500)

# subset the labels and features
train_message = df_train['final_message'].tolist()
train_message = df_test['final_message'].tolist()
train_label = df_train['salary_bucket'].tolist()
test_label = df_test['salary_bucket'].tolist()

# vectorize the dataset
vectorizer = CountVectorizer()
tfidf = TfidfTransformer()
vectorizer.fit(train_message)
train_matrix = vectorizer.transform(train_message)
test_matrix = vectorizer.transform(train_message)
tfidf.fit(train_matrix)
tfidf_train = tfidf.transform(train_matrix)
tfidf_test = tfidf.transform(test_matrix)

# define function for generating confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Model implementation - Bernoulli Naive Bayes
bernoulli = BernoulliNB()
bernoulli.fit(tfidf_train, train_label)                  # fit
predict_bernoulli = bernoulli.predict(tfidf_test)        # predict

# print out the accuracy score
print ('Bernoulli accuracy score:', accuracy_score(test_label, predict_bernoulli))

#-----------------------------------------------------------------------------
# Bernoulli accuracy score 0.826
#-----------------------------------------------------------------------------

# plot the confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(confusion_matrix(test_label, predict_bernoulli), classes=class_names,
                      title='Bernoulli Confusion matrix')
plt.show()

# Model implementation - Multinomial Naive Bayes
multinomial = MultinomialNB()
multinomial.fit(tfidf_train, train_label)                  # fit
predict_multinomial = multinomial.predict(tfidf_test)      # predict

# print out the accuracy score
print ('Multinomial accuracy score:',accuracy_score(test_label, predict_multinomial))

#-----------------------------------------------------------------------------
# Multinomial accuracy score: 0.772
#-----------------------------------------------------------------------------

# plot the confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(confusion_matrix(test_label, predict_multinomial), classes=class_names,
                      title='Multinomial Confusion matrix')
plt.show()

"""
                    Ways to Improve Accuracy of Model
    
At a very high level, the best way to improve the performance of a text classification
model is to select the right features 
- choosing relevant features and thinking about how to encode them can have 
a great impact on the learning method's ability to extract and fit a good model

- get creative with identifying patterns and extracting features from a corpus
-> examples of this could include creating a regular expression tagger that 
chooses a part of speech tag for a particular word based upon the internal makup
of that word

- looking at the context of a word to extract features based on its associated words
-> attach meaning for particular words, such as whether the word bank refers to 
a financial institution or a river bank   

- create joint classifier models to select appropriate labelling for a series 
of related inputs
-> this would essentially capture the dependencies between related classification
tasks 
-> an example of this method in practice is the greedy sequence classification 
strategy which initially finds a class label for the first input and recursively 
builds on that to help determine the most appropriate label for the next input
-> to build on this, Hidden Markov Models further expand on the greedy sequence 
classification framework by integrating a transformational strategy which generate
a probability distribution over tags 

- using different packages and models could help define performance metrics 
accordingly
-> the sklearn package have Bernoulli, Gaussian and Multinomial Naive Bayes functions
which have embedded hyperparameter tuning mechinisms to optimize performance  

- other considerations could involve engineering features from the other attributes
in the dataset
-> combining supervised and unsupervised methods 
--> salary can be computed by location and averaged out across all observations for
that region and then can implement k-means cluserting to assign labels for newly 
engineering feature
--> this new feature can be used to compare values of actual salary vs the regional
difference and then can be fit into another set to look and contrast top features across 
both sets

Overall, it requires a great amount of effort to examine each dataset carefully 
to determine an appropriate feature extractor for each use case
"""
