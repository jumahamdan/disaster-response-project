# import packages
import sys

# import libraries
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
import pickle

# nltk
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# classifiers
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# metrics
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import label_ranking_average_precision_score

# visuals
import itertools
import matplotlib.pyplot as plt

nltk.download(['punkt','stopwords','wordnet'])

def load_data(database_filepath = 'sqlite:///DisasterResponse.db'):
    '''
    Function: Load data from database and return X and y
    Args:
        db_path(str): path of database file name
    Return:
        X: messages for training set
        y: labels of messages for test set
    '''
    # load data from database
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('labeled_messages' ,engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    Function: tokenize the text
    Args:  source string
    Return:
    clean_tokens(str list): clean string list
    
    """
    #normalize text
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
    
    #token messages
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]
    
    #sterm and lemmatizer
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens  


def build_model():
    ''' build a machine learning model '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200,random_state=20)))
    ])
    
    parameters = {
        'clf__estimator__criterion':['entropy']
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)

    return model_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    ''' show the accuracy, precision, and recall of the tuned model'''
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test.columns.values):
        accuracy = accuracy_score(Y_test.loc[:,col], Y_pred[:,i])
        print(f'Feature {i+1}: {category_names[i]} Accuracy: {100*accuracy:.2f}% \n')
        print(classification_report(Y_test[col], Y_pred[:,i]))


def save_model(model, model_filepath):
    ''' export the model as a pickle file '''
    pickle.dump(model, open(model_filepath, 'wb')) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()