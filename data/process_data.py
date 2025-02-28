"""ETL pipeline for loading, cleaning & saving data from csv files to a sqlite database."""
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_messages_categories_from_csv(messages_filepath, categories_filepath) -> pd.DataFrame:
    """Load & merge messages & categories datasets from csv files.
    
    Args:
        messages_filepath (str) Filepath for csv file containing messages dataset.
        categories_filepath (string) Filepath for csv file containing categories dataset.
       
    returns:
        df (pd.Dataframe) Dataframe containing merged content of messages & categories datasets.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'left', on = ['id'])
    return df

def clean_data(df_raw) -> pd.DataFrame:
    """Clean dataframe by removing duplicates & converting categories from strings 
    to binary values.
    
    Args:
        df_raw (pd.Dataframe): A dataframe containing merged content of messages 
        and categories datasets.
        The steps are as follows:
        1. Split categories into separate category columns.
        2. Convert category values to just numbers 0 or 1.
        3. Replace categories column in df with new category columns.
        4. Remove duplicates.
        5. Remove rows with a value of 2 from the column related.

    Returns:
        df_clean (pd.Dataframe): A cleaned version of the input dataframe. 
        
    """
    categories = df_raw['categories'].str.split(pat=';',expand=True)
    row = categories.iloc[0]
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].transform(lambda x: x[-1:])
        categories[column] = pd.to_numeric(categories[column])

    df_clean = df_raw.copy(deep=True)
    df_clean.drop('categories', axis = 1, inplace = True)
    df_clean = pd.concat([df_clean, categories], axis = 1)
    df_clean.drop_duplicates(inplace = True)
    df_clean = df_clean[df_clean['related'] != 2]
    return df_clean

def save_data_to_database(df, database_filename) -> None:
    """This function saves the clean dataset into an sqlite database.

    Args:
        df (pd.Dataframe): The clean dataset to be saved
        database_filename (str): The filename for the database (including filepath)
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')

def main() -> None:
    """This function executes the ETL pipeline.
        It loads the messages & categories datasets, cleans the data,
        and then saves it to a sqlite database.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df_raw = load_messages_categories_from_csv(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df_clean = clean_data(df_raw= df_raw)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data_to_database(df_clean, database_filepath)

        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages & categories '\
              'datasets as the first & second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
