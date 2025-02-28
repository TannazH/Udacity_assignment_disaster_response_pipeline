"""This module contains the functions for ML pipeline, including loading
data from database, tokenizing text data, building the model pipeline, 
training the model, evaluating the model and saving the model and evaluation matrix.
"""
import re
import sys
from termcolor import colored
import pickle
import warnings
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn import preprocessing

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab', 'averaged_perceptron_tagger_eng'])
le = preprocessing.LabelEncoder()
warnings.simplefilter('ignore')


def load_data_from_database(database_filepath) -> tuple:
    """
    This function loads the data from the database and returns the features, labels and category names.

    Args:
        database_filepath: string. Filepath for sqlite database containing cleaned message data.

    Returns:
        x_features (pd.Dataframe): Dataframe containing features.
        y_labels (pd.Dataframe): Dataframe containing labels.
        category_names (list) List containing category names.
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    x_features = df['message']
    y_labels = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(np.array(y_labels.columns))
    return x_features, y_labels, category_names


def tokenize(text) -> list:
    """This function tokenizes the text data and returns a clean token list 
        using the following steps:
        1. Convert text to lower case and remove punctuation
        2. Tokenize text
        3. Lemmatize the word tokens
        4. Remove stop words
        5. Return the clean tokens

    Args:
        text (str): A string containing the message data.

    Returns:
        clean_tokens (list): List containing clean tokens.
    """
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    stop_words = stopwords.words("english")
    tokens = word_tokenize(text)
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens if w not in stop_words]

    return clean_tokens

def build_model() -> GridSearchCV:
    """This function builds the model pipeline using the following steps:
        1. CountVectorizer: Convert a collection of text documents to a matrix of token counts.
        2. TfidfTransformer: Transform a count matrix to a normalized tf or tf-idf representation.
        3. MultiOutputClassifier(RandomForestClassifier): Multi target classification using the RandomForestClassifier,
        where RandomForestClassifier is an ensemble learning method for classification.

        parameters of the pipeline are optimized using GridSearchCV,
        this parameters include:
        - vect__min_df: Ignore terms that have a document frequency strictly 
        lower than the given threshold.
        - tfidf__use_idf: Enable inverse-document-frequency reweighting.
        - clf__estimator__n_estimators: The number of trees in the forest.
        - clf__estimator__min_samples_split: The minimum number of samples required 
        to split an internal node.

    Returns:
        cv (GridSearchCV): GridSearchCV object.
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'vect__min_df': [1, 5],
                'tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators': [10, 25],
                'clf__estimator__min_samples_split': [2, 4]}
    
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, x_test, y_test, category_names, print_option:bool=True) -> pd.DataFrame:
    """ This function evaluates the model and prints the precision, recall and f1-score for each output category.

    Args:
        model: model object. Trained model.
        x_test (pd.Dataframe): Test features.
        y_test (pd.Dataframe): Test labels.
        category_names: (list) List containing category names.

    Returns:
        evaluation_matrix (pd.Dataframe): Dataframe containing precision, recall and f1-score for
        each output category.
    """
    y_pred = model.predict(x_test)
    evaluation_matrix = pd.DataFrame(columns=['precision', 'recall', 'f-score'],
                                     index=category_names)
    precision = []
    recall = []
    fscore = []
    for i, col in enumerate(category_names):
        precision_col, recall_col, fscore_col, _ = precision_recall_fscore_support(y_test[col],
                                                                       y_pred[:, i],
                                                                       average='weighted')
        if print_option:
            print("\nReport for the column ({}):\n".format(colored(col, 'red', attrs=['bold', 'underline'])))

            if precision_col >= 0.80:
                print('Precision: {}'.format(colored(round(precision_col, 2), 'green')))
            else:
                print('Precision: {}'.format(colored(round(precision_col, 2), 'red')))

            if recall_col >= 0.80:
                print('Recall: {}'.format(colored(round(recall_col, 2), 'green')))
            else:
                print('Recall: {}'.format(colored(round(recall_col, 2), 'red')))

            if fscore_col >= 0.80:
                print('F-score: {}'.format(colored(round(fscore_col, 2), 'green')))
            else:
                print('F-score: {}'.format(colored(round(fscore_col, 2), 'red')))
        

        precision.append(precision_col)
        recall.append(recall_col)
        fscore.append(fscore_col)
    evaluation_matrix['precision'] = precision
    evaluation_matrix['recall'] = recall
    evaluation_matrix['f-score'] = fscore

    return evaluation_matrix

def save_model(model, model_filepath) -> None:
    """
    Save model to a pickle file

    Args:
        model (model object) Trained model.
        model_filepath (string): Filepath to save the model. 
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def save_evaluation_matrix(evaluation_matrix, evaluation_filepath) -> None:
    """
    Save evaluation matrix to a pickle file

    Args:
        evaluation_matrix (pd.Dataframe): Dataframe containing precision, recall and f1-score for
        each output category.
        evaluation_filepath (string): Filepath to save the evaluation matrix.
    """
    evaluation_matrix.to_csv(evaluation_filepath, index=True)

def main():
    if len(sys.argv) == 4:
        database_filepath, model_filepath = sys.argv[1:3]
        evaluation_filepath = sys.argv[3]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        x_features, y_labels, category_names = load_data_from_database(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(x_features, y_labels, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluation_matrix = evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    model: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

        print('Saving evaluation matrix...\n    evaluation matrix: {}'.format(evaluation_filepath))
        save_evaluation_matrix(evaluation_matrix, evaluation_filepath)

        print('Evaluation matrix saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument and the filepath to the csv file for the evaluation matrix.'\
              '. \n\nExample: python '\
              'models/train_classifier.py data/DisasterResponse.db models/classifier.pkl models/evaluation_matrix.csv')
        
if __name__ == '__main__':
    main()

