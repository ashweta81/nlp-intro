# --------------
# Importing Necessary libraries
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the 20newsgroups dataset
df = fetch_20newsgroups(subset='train')
pprint(list(df.target_names))

#Create a list of 4 newsgroup and fetch it using function fetch_20newsgroups
ctgy = ['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware']
df_train = fetch_20newsgroups(subset='train', categories=ctgy)

#Use TfidfVectorizer on train data and find out the Number of Non-Zero components per sample.
tfvec = TfidfVectorizer()
tftrain = tfvec.fit_transform(df_train.data)
print(tftrain.shape)

#Use TfidfVectorizer on test data and apply Naive Bayes model and calculate f1_score.
newsgroup_test = fetch_20newsgroups(subset='test', categories=ctgy)
tftest = tfvec.transform(newsgroup_test.data)

nb = MultinomialNB()
nb.fit(tftrain, df_train.target)
pred = nb.predict(tftest)
print('The f1 score is', f1_score(newsgroup_test.target, pred, average='macro'))
#Print the top 20 news category and top 20 words for every news category




