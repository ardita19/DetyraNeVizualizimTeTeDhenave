import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
data = pd.read_csv("PreProcessedFinal1.csv", index_col=0)
vect = CountVectorizer()
nb = MultinomialNB()
# nb = svm.SVC()
# nb = DecisionTreeClassifier()
# nb = KNeighborsClassifier(n_neighbors=3)
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer()

X_train, X_test, y_train, y_test = train_test_split(data['ReviewBody'].astype('U'), data['sentiment'].astype('U'), test_size=0.4, random_state=30)

print(X_train)

X_train_set = vect.fit_transform(X_train)
X_test_set = vect.transform(X_test)

# svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
# svd.fit_transform(X_train_set)
# TruncatedSVD(n_components=5, n_iter=7, random_state=42)
# print(svd.explained_variance_ratio_)
# print(svd.explained_variance_ratio_.sum())
# print(svd.singular_values_)
# normalizer = Normalizer(copy=False)
# lsa = make_pipeline(svd, normalizer)
# X_train_set = lsa.fit_transform(X_train_set)
# X_test_set = lsa.fit_transform(X_test_set)

nb.fit(X_train_set, y_train)
y_pred = nb.predict(X_test_set)
precision, recall, fscore, support = precision_recall_fscore_support(y_test,y_pred,average='macro'  )
print(accuracy_score(y_test, y_pred))
print(precision)
print(recall)
print(fscore)




