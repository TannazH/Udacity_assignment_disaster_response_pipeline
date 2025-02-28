# Udacity_assignment_disaster_response_pipeline

# Disaster Response Pipeline Project

## Installation

for using this package 
1. Create and activate a virtual environment: open your terminal and run the following commands:

    python -m venv myenv

2. Activate the virtual environment

**Windows:** 

myenv\Scripts\activate

**macOS and Linux:**

source myenv/bin/activate                                               

This repository requires the following Python packages: 


    blinker==1.9.0
    click==8.1.8
    colorama==0.4.6
    Flask==3.1.0
    greenlet==3.1.1
    itsdangerous==2.2.0
    Jinja2==3.1.5
    joblib==1.4.2
    MarkupSafe==3.0.2
    narwhals==1.28.0
    nltk==3.9.1
    numpy==2.2.3
    packaging==24.2
    pandas==2.2.3
    plotly==6.0.0
    python-dateutil==2.9.0.post0
    pytz==2025.1
    regex==2024.11.6
    scikit-learn==1.6.1
    scipy==1.15.2
    six==1.17.0
    SQLAlchemy==2.0.38
    termcolor==2.5.0
    threadpoolctl==3.5.0
    tqdm==4.67.1
    typing_extensions==4.12.2
    tzdata==2025.1
    Werkzeug==3.1.3


## Project Overview
The objective of this assignment is to deign a web app which can be used by emergency operators during a disaster to classify a disaster text messages into several categories which then can be transmited to the responsible entity.

The app built to have an ML model to categorize every message received

## File Description:
* **process_data.py**: This python excutuble code takes as its input csv files containing message data and message categories (labels), and then creates a SQL database
* **train_classifier.py**: This code trains the ML model with the SQL data base
* **data**: This folder contains sample messages and categories datasets in csv format.
* **app**: cointains the run.py to iniate the web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and save the pickle file and the evaluation matrix output 
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl models/evaluation_matrix.csv`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001
   Or Go to http://192.168.178.97:3001

## Figures

***Figure 1: App taining dataset***
![Screenshot 1](https://github.com/TannazH/Udacity_assignment_disaster_response_pipeline/tree/main/figures/overview_training_dataset.png)


***Figure 2: App classification search page***
![Screenshot 2](https://github.com/TannazH/Udacity_assignment_disaster_response_pipeline/tree/main/figures/message_classification.png)


## Licensing, Authors, Acknowledgements
This project was completed as part of the [Udacity Data Scientist Nanodegree].
