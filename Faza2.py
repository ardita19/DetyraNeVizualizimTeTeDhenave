import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
import json
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer

data = pd.read_csv("PreProcessedFinal1.csv", index_col=0)
# data['number_of_words'] = data.apply(lambda x: x['ReviewBody'].size(), axis=1)

#Mode
print("mode: ", data['ReviewStar'].mode())
print(data.groupby('ReviewStar').size())

#Median
print("median", data['ReviewStar'].median())

#Mean

print("stars mean: ", data['ReviewStar'].mean())
print("price mean: ", data['Price'].mean())

#Range
print("max", data['Price'].min())
print("min", data['Price'].max())
print("range", data['Price'].max() - data['Price'].min())

#Percentile

print("Percentile", np.percentile(data['Price'], 70))

#Variance
print("variance: ", data['Price'].var())

#StandardDeviation
print("standard deviation: ", data['Price'].std())

count_v = CountVectorizer()
word_count_matrix = count_v.fit_transform(data[data['ReviewStar'] >= 3]['ReviewBody'].astype('U'))


count_list = word_count_matrix.toarray().sum(axis=0)
word_list = count_v.get_feature_names()
#print(word_list)
word_freq = pd.DataFrame(count_list, index=count_v.get_feature_names(), columns=['Freq'])
word_freq = word_freq.sort_values("Freq", ascending=False)
#print(word_freq)

#Frekuenca
word_freq['frekuencaNe%'] = word_freq.apply(lambda x: (x['Freq']/word_freq.sum())*100, axis=1)
print(word_freq)
word_freq.sort_values(by='Freq', ascending=False).head(30)

count_v = CountVectorizer()
word_count_matrix = count_v.fit_transform(data['ReviewBody'].astype('U'))

count_list = word_count_matrix.toarray().sum(axis=0)
nrOfWordsPerTweet = word_count_matrix.toarray().sum(axis=1)
plt.scatter(data['Price'], nrOfWordsPerTweet)
plt.show()

# plt.scatter(data['ReviewStar'], nrOfWordsPerTweet)
# plt.show()


#Outliers
plt.boxplot(word_freq['frekuencaNe%'].head(50))
plt.show()

#Menjanimi i zbulimeve jo te sakta
data['ReviewBody'] = data['ReviewBody'].map(lambda x: x.replace("'sound'",""))
data['ReviewBody'] = data['ReviewBody'].map(lambda x: x.replace("'bass'",""))
data['ReviewBody'] = data['ReviewBody'].map(lambda x: x.replace("'product'",""))
print(data['ReviewBody'][0])

#print(word_count_matrix.toarray())