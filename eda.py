import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import pandas_profiling as pp

df = pd.read_csv('dataset/train.csv', encoding='latin-1')

report = pp.ProfileReport(df)
report.to_file('outputs/profile_report.html')

# Extracting hashtags from text
df['hashtags'] = df['text'].apply(lambda x:re.findall('#\w*',x))

df.head()

# WordCloud Visualization
from wordcloud import WordCloud
labels = ['Negative', 'Positive']
no_clusters=2
for c in range(2):
    hts=list(df[df['target']==c]['hashtags'])
    
    hashes = []
    for ht in hts:
        for h in ht:
            hashes.append(h.strip())
            
    string_hash=' '.join(hashes)
    hash_values=pd.Series(hashes).value_counts()
    hval=hash_values.reset_index()
    
    #wordcloud plot
    d = {}
    for a, x in hval.values:
        d[a] = x
    
    wordcloud = WordCloud(max_font_size=40)
    wordcloud.generate_from_frequencies(frequencies=d)
    plt.figure(figsize=(70,70))
    
    if not c:
        plt.imsave('outputs/wordcloud_negative.png', wordcloud)
    else:
        plt.imsave('outputs/wordcloud_positive.png', wordcloud)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()