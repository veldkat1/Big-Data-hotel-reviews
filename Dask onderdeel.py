import dask.dataframe as dd
import pandas as pd 
from dask.dataframe import from_pandas
from dask_ml.model_selection import train_test_split as train_test_split_d
from sklearn.model_selection import train_test_split as train_test_split_s
from dask_ml.metrics import accuracy_score as accuracy_score_d
from sklearn.metrics import accuracy_score as accuracy_score_s
from dask_glm.datasets import make_classification
from dask_ml.linear_model import LogisticRegression as DaskLogisticRegression
from sklearn.linear_model import LogisticRegression as scikitLogisiticRegression
from dask_glm.datasets import make_classification
from dask import distributed
import dask.array as da
import dask.delayed as delayed
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import joblib
import time
from nltk.corpus import stopwords 
nltk.download('stopwords')
from MongoDatabase_onderdeel import laad_data
from dask.distributed import Client, progress

def tfidf(df):
    vect = TfidfVectorizer(ngram_range=(1, 2), 
                            max_features=500, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df['review'].values.astype('U'))
    X = vect.transform(df['review'].values.astype('U'))
    X_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
    return X_df

def scikit_logistic_regression(df):
    print("Logisitic regression zonder dask")
    t0 = time.time()
    df=df
    X = tfidf(df)
    y = df[ 'label']
    X_train, X_test, y_train, y_test = train_test_split_s(X, y, test_size=0.2, random_state=42)

    lr_scikit = scikitLogisiticRegression(max_iter=10000,C=10000.0, penalty='l2')
    lr_scikit.fit(X_train, y_train)
    y_pred_lr = lr_scikit.predict(X_test)
    accuracy_lr = accuracy_score_s(y_test, y_pred_lr)
    t1 = time.time()
    model_path = (r'C:\Users\thijs\Desktop\assignment 2\opgeslagen dask en scikit-learn modellen\scikit_model.pkl')
    joblib.dump(lr_scikit, model_path)
    return lr_scikit, accuracy_lr, t1 - t0

def dask_logistic_regression(df):
    print("Logistic_regression met dask")
    t0 = time.time()

    X = tfidf(df)
    y = df[ 'label']
    X, y = make_classification()
    lr_dask = DaskLogisticRegression(max_iter=10000,C=10000.0, penalty='l2')

    lr_dask.fit(X, y)
    lr_dask.decision_function(X)
    lr_dask.predict(X)
    lr_dask.predict_proba(X)
    accuracy_dask = lr_dask.score(X, y)
    t1 = time.time()
    model_path = (r'C:\Users\thijs\Desktop\assignment 2\opgeslagen dask en scikit-learn modellen\dask_model.pkl')
    joblib.dump(lr_dask, model_path)
    return lr_dask, accuracy_dask, t1 - t0

if __name__ == "__main__":
    df = laad_data("cleaned_data")
    tfidf_result = tfidf(df)
    lr_scikit, accuracy_lr, training_time_scikit = scikit_logistic_regression(df)
    print("Scikit-Learn Logistic Regression Accuracy:", accuracy_lr)
    print("Training Time (seconds):", training_time_scikit)
    lr_dask, accuracy_dask, training_time_dask = dask_logistic_regression(df)
    accuracy_dask = accuracy_dask.compute()
    print("dask Logistic Regression Accuracy:", accuracy_dask)
    print("Training Time (seconds):", training_time_dask)