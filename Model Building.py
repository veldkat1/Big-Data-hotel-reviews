import pandas as pd
import mysql.connector
import nltk
import matplotlib.pyplot as plt
import joblib
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from dask_ml.linear_model import LogisticRegression as LogisticRegressionDask
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgb
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

csv_bestand = r'C:\Users\thijs\Desktop\assignment 1/cleaned_data.csv'

# DataFrame inlezen met een puntkomma als scheidingsteken
df = pd.read_csv(csv_bestand, sep=',')

# Het ingelezen DataFrame bekijken
print("dataframe is ingeladen")

# opnieuw toevoegen van stopwoorden.

stopwords_en = set(stopwords.words('english'))
stopwords_nl = set(stopwords.words('dutch'))
custom_stopwords = ['hotel', 'room','rooms','staff','breakfast','bathroom','hotel','kamer','kamers','medwerksrs','ontbijt','badkamer','could','would','get']
combined_stopwords = list(stopwords_en.union(stopwords_nl).union(custom_stopwords))
en_nl = list(stopwords_en.union(stopwords_nl))

#instellen van count_vectorizer

def count_vectorizer(df, combined_stopwords):
    vect = CountVectorizer(stop_words=combined_stopwords, ngram_range=(1, 2), 
                             max_features=1000, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df['review'])
    X = vect.transform(df['review'])
    X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
    joblib.dump(vect, r'C:\Users\thijs\Desktop\assignment 1\vectorizers\count_vectorizer.joblib')
    return X_df

def count_vectorizer_stem(df, combined_stopwords):
    vect = CountVectorizer(stop_words=combined_stopwords, ngram_range=(1, 2), 
                             max_features=1000, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df['stemmed_text'])
    X = vect.transform(df['stemmed_text'])
    X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
    return X_df

def count_vectorizer_lem(df, combined_stopwords):
    vect = CountVectorizer(stop_words=combined_stopwords, ngram_range=(1, 2), 
                             max_features=1000, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df['lemmatized_text'])
    X = vect.transform(df['lemmatized_text'])
    X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
    return X_df


# Maak vectorizers voor individuele teksten
X_df = count_vectorizer(df, combined_stopwords)
X_dfs = count_vectorizer_stem(df, combined_stopwords)
X_dfl = count_vectorizer_lem(df, combined_stopwords)


#Het instellen van het logistic regression model.
def logistic_regression(X_df):
    X = X_df
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_classifier = LogisticRegression(max_iter=5000, random_state=0, solver='lbfgs')
    lr_classifier.fit(X_train, y_train)

    y_pred = lr_classifier.predict(X_test)
    y_pred_prob = lr_classifier.predict_proba(X_test)[:, 1]  # krijg de waarschijnlijkheid van de positieve klasse


    # Bereken de nauwkeurigheid en print het classification report
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    confusion = confusion_matrix(y_test, y_pred)
    print("confusion matrix", confusion)

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print("ROC_AUC", roc_auc)

    joblib.dump(lr_classifier, r'C:\Users\thijs\Desktop\assignment 1\models\logistic_regression_model.joblib')

logistic_regression(X_df)
print("logistic_regression klaar")

#het instellen van het naive bayes model
def naive_bayes(X_df):
    X = X_df
    y = df['label']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)

    y_pred = nb_classifier.predict(X_test)
    y_pred_prob = nb_classifier.predict_proba(X_test)[:, 1]  # Get the probability of positive class

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

    confusion = confusion_matrix(y_test, y_pred)
    print("confusion matrix", confusion)

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print("ROC_AUC", roc_auc)

    joblib.dump(nb_classifier, r'C:\Users\thijs\Desktop\assignment 1\models\naive_bayes_model.joblib')

naive_bayes(X_df)
print("Naive Bayes klaar")

# het instellen van het random forest model
def random_forest(X_df):
    X = X_df
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=50, n_jobs=5, max_depth = None)
    
    rf_classifier.fit(X_train, y_train)
    
    y_pred = rf_classifier.predict(X_test)
    y_pred_prob = rf_classifier.predict_proba(X_test)[:, 1]  # Get the probability of positive class
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    report = classification_report(y_test, y_pred)
    print("Classification Report:", report)

    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:", confusion)

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print("ROC AUC Score:", roc_auc)

    joblib.dump(rf_classifier, r'C:\Users\thijs\Desktop\assignment 1\models\random_forest_model.joblib')

random_forest(X_df)

print("Random forrest klaar")

loaded_count_vectorizer = joblib.load(r'C:\Users\thijs\Desktop\assignment 1\vectorizers\count_vectorizer.joblib')

logistic_regression_model = joblib.load(r'C:\Users\thijs\Desktop\assignment 1\models\logistic_regression_model.joblib')
naive_bayes_model = joblib.load(r'C:\Users\thijs\Desktop\assignment 1\models\naive_bayes_model.joblib')
random_forest_model = joblib.load(r'C:\Users\thijs\Desktop\assignment 1\models\random_forest_model.joblib')
