import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import pandas_profiling as pp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from preprocess import Process

df = pd.read_csv('dataset/train.csv', encoding='latin-1')

report = pp.ProfileReport(df)
report.to_file('outputs/profile_report.html')

# Extracting hashtags from text
df['hashtags'] = df['text'].apply(lambda x:re.findall('#\w*',x))
  
df['processed_text'] = df.tweet.apply(lambda row : Process().preprocess_text(row))
df.head()

tfidf_vec = TfidfVectorizer(ngram_range=(1,3))
tfidf_data = tfidf_vec.fit_transform(df.processed_text)
#tfidf_data = pd.DataFrame(tfidf_data).toarray()
#tfidf_data.head()

X_train, X_test, y_train, y_test = train_test_split(tfidf_data, df['label'], test_size=0.0, random_state = 42)

spam_filter = MultinomialNB(alpha=0.2)
spam_filter.fit(X_train, y_train)



# ------------------------Testing-------------------------------

predictions = spam_filter.predict(X_test).tolist()
wrong = []
count = 0
for i in range(len(y_test)):
    if y_test.iloc[i] != predictions[i]:
        count += 1
        wrong.append(i)

      
print('Total number of test cases', len(y_test))
print('Number of wrong of predictions', count)

from sklearn.metrics import classification_report
print(classification_report(predictions, y_test))