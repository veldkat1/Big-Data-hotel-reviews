import pandas as pd
import mysql.connector
import nltk
import os
import timeit
import wordnet
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from nltk.stem import SnowballStemmer 
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

#Maken van connectie met database
db_conn = mysql.connector.connect(
    host="oege.ie.hva.nl",
    user="veldkat1",
    password="+#$pPXlpGtwZVM",
    database="zveldkat1")
if db_conn.is_connected():
    print("succesfully connected")
else:
    print("failed connection")

cursor = db_conn.cursor()

#Maken van queries voor datacleaning
query1 = "SET SQL_SAFE_UPDATES = 0;"
query2 = """DELETE FROM hotel_reviews
	WHERE review = 'No Negative' OR review = 'No Positive' OR review like 'No Negative' OR review like 'No Positive' 
    OR review like 'Nothing' OR review like 'Not a thing' 
    OR review like 'nothing' OR review like 'NA' OR review like 'none' OR review like 'Everything'
    OR review like 'NOTHING' OR review like 'EVERYTHING' OR review like 'everything';"""
query3 = "SET SQL_SAFE_UPDATES = 1;"

#Uitvoeren van de queries
try:
    cursor.execute("BEGIN")
    cursor.execute(query1)
    cursor.execute(query2)
    cursor.execute(query3)
    cursor.execute("COMMIT")

    print("Queries uitgevoerd en gecommit.")

    #gegevens ophalen in dataframe
    query4 = "SELECT * FROM `hotel_reviews`"
    df = pd.read_sql(query4, con=db_conn)
    print("gegevens in dataframe geladen")


except mysql.connector.Error as error:
    #rollback in geval van fout.
    cursor.execute("ROLLBACK")
    print("Fout bij het uitvoeren van queries.", error)


cursor.close()
db_conn.close

#cleaning voor het weggooien van bepaalde rows
def cleaning(df):
    df["review"] = df["review"].apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))
    df["review"] = df['review'].str.lower()
    df['review'] = df['review'].str.replace('\d+', '', regex=True)
    df['Aantal Woorden'] = df['review'].str.split().apply(len)
    df= df.loc[df['Aantal Woorden'] >= 5]
    df = df.drop('Aantal Woorden', axis=1)
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

df = cleaning(df)
print("Cleaning klaar")

save_dir = (r"C:\Users\thijs\Desktop\assignment 1\wordclouds")
#laden van Nederlandse en Engelse stopwoorden
stopwords_en = set(stopwords.words('english'))
stopwords_nl = set(stopwords.words('dutch'))
custom_stopwords = ['hotel', 'room','rooms','staff','breakfast','bathroom','hotel','kamer','kamers','medwerksrs','ontbijt','badkamer','could','would','get']
combined_stopwords = list(stopwords_en.union(stopwords_nl).union(custom_stopwords))
en_nl = list(stopwords_en.union(stopwords_nl))


#maken van de wordclouds
def create_wordcloud(df, save_dir):
    totaal_cloud = WordCloud(background_color='white', stopwords=en_nl).generate(' '.join(df['review']))
    plt.imshow(totaal_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, 'totaal_cloud.png'))

    df_positief = df[df['label'] == 1]
    cloud_positief = WordCloud(background_color='white', stopwords=combined_stopwords).generate(' '.join(df_positief['review']))
    plt.imshow(cloud_positief, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, 'cloud_positief.png'))

    df_negatief = df[df['label'] == 0]
    cloud_negatief = WordCloud(background_color='white', stopwords=combined_stopwords).generate(' '.join(df_negatief['review']))
    plt.imshow(cloud_negatief, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, 'cloud_negatief.png'))

create_wordcloud(df, save_dir)


# Initialiseren van SnowballStemmer voor Engels en Nederlands
stemmer_en = SnowballStemmer('english')
stemmer_nl = SnowballStemmer('dutch')

# Stemming toepassen op de review kolom
df['stemmed_text'] = df['review'].apply(lambda x: ' '.join([stemmer_en.stem(word) if word.isalpha() else word for word in x.split()]))
df['stemmed_text'] = df['stemmed_text'].apply(lambda x: ' '.join([stemmer_nl.stem(word) if word.isalpha() else word for word in x.split()]))

print("stemming klaar")


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith("N"):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# Initialiseren van WordNetLemmatizer voor Engels
lemmatizer_en = WordNetLemmatizer()
lemmatizer_nl = WordNetLemmatizer()

# POS-tagging en lemmatisering toepassen op de review kolom
def apply_lemmatization(text):
    tokens = nltk.word_tokenize(text)  # Tokenize de tekst
    tagged_tokens = nltk.pos_tag(tokens)  # Voer POS-tagging uit op de tokens

    lemmatized_tokens = []
    for token, tag in tagged_tokens:
        pos = pos_tagger(tag)  # Krijg de POS-tag voor het woord
        if pos is not None:
            lemmatized_word = lemmatizer_en.lemmatize(token, pos=pos) if token.isalpha() else token
        else:
            lemmatized_word = lemmatizer_en.lemmatize(token) if token.isalpha() else token
        lemmatized_tokens.append(lemmatized_word)

    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

df['lemmatized_text'] = df['review'].apply(apply_lemmatization)
print("Lemmatisering klaar")

#maken van een nieuwe csv file
Folderpath = (r"C:\Users\thijs\Desktop\assignment 1") 
csv_path = os.path.join(Folderpath,"cleaned_data.csv")
df.to_csv(csv_path, index=False)

