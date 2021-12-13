import pandas as pd
import emoji
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import spacy
import nltk
from string import punctuation
import re
import csv
def remove_punctuation(text):
    return ''.join(char for char in text if char not in punctuation)

def remove_stopwords(text):
    text_tokens = word_tokenize(text)
    return [word for word in text_tokens if not word in stopwords.words('english')]

def lemmatisation(text):

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(u''+text)
    tokens = []
    for token in doc:
        tokens.append(token)

    lemmatized_sentence = " ".join([token.lemma_ for token in doc])
    return (lemmatized_sentence)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def removeRepeatedLetters(text):
    x = re.split("\s", text)
    for i in range(0,len(x)):
        for j in range(97,122):

            x[i]=re.sub(chr(j)+'{3,}',chr(j)+chr(j),x[i])
        for k in range(65, 90):

            x[i] = re.sub(chr(k) + '{2,}', chr(k)+chr(k), x[i])
    return " ".join(x)

def convert_emojis_to_words(text):
    text = emoji.demojize(text)
    text = text.replace(":", " ")
    text = ' '.join(text.split())

    text = [s.replace('_', ' ') for s in text]

    return remove_punctuation(text)

def abbreviation_to_words(text):
    j = 0
    for string in text:
        fileName = "D:\\slang.txt"
        with open(fileName) as myCSVfile:
            dataFromFile = csv.reader(myCSVfile, delimiter="=")
            string = re.sub('[^a-zA-Z0-9-_.]', '', string)
            for row in dataFromFile:
                if string.upper() == row[0]:
                    text[j] = row[1].lower()
            myCSVfile.close()
        j = j + 1
    return remove_punctuation(" ".join(text)).split(" ")

data = pd.read_csv('MergeFile3.csv', index_col=0)
print(data)
reviews  = []
for count in range(len(data)):
    review = data.at[count, 'ReviewBody']
    print(review)
    initialreview=review
    review = "".join(review)
    lemmatizer = WordNetLemmatizer()
    review = re.sub('@([A-Za-z0-9_]+)', '', review)
    # review1=re.sub('[\s]+','',review1)
    review = re.sub(r'#([^\s]+)', r'', review)
    review = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', review)
    review = re.sub(r'\w*\d\w*', '', review)

    review = convert_emojis_to_words(review)
    review = removeRepeatedLetters(review)
    review = " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(lemmatisation(review))])
    review = review.replace('-PRON-', '')
    review = remove_stopwords(review.lower())
    review = abbreviation_to_words(review)
    indexNames11 = data[data['ReviewBody'] == initialreview].index
    data.at[count, 'ReviewBody'] = review
    print(review)
    print(count)

data.to_csv("Pre-processed8.csv")


