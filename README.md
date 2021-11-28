# Health Insurance Cross Sell Prediction

Small project made as an exercise in ML. It aims to predict health insurance owners who will be interested in vehicle insurance based on a <a href="https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction">Kaggle dataset</a>. Utilises NumPy, pandas and scikit-learn tools. Comes with a Web API made with Flask, and a Dockerfile for containerization.

## Problem Statement

An insurance company would like to plan its communication strategy to reach the customers who are willing to purchase Vehicle Insurance. Thus, the company will optimise its business model and revenue. In order to do so, it is highly helpful to predict whether the policyholders of Health Insurance from the past year will also be interested in Vehicle Insurance. The objective is to build a model for making such predictions. The provided data contains information about demographics (gender, age, region), vehicles (vehicle age, vehicle damage), policy (premium, sourcing channel) etc.

## Solution

The customers’ willingness to buy Vehicle Insurance (‘Yes’ or ‘No’ response) is predicted with the accuracy of 0.84. The prediction is made with the Gradient Boosting Classifier (GBC), which appeared to be among the preselected estimators with the highest accuracy score, and exhibited relatively high robustness to changes in the dataset. Cross-validation was performed with a stratified shuffle split.  

The models' accuracy comparison:

<p align="center">
  <img src="./images/acc_1.png">
</p>

<p align="center">
  <img src="./images/acc_2.png">
</p>

The GBC performance:

<p align="center">
  <img src="./images/class_report_GBC.png">
</p>

<p align="center">
  <img src="./images/GBC_ROC.png">
</p>  
  
The accuracy score for GBC on the test dataset is 0.8448.


## Technologies

- Python 3.8.8
- Flask 2.0.2
- Werkzeug 2.0.2
- pandas 1.3.4
- NumPy 1.21.3
- scikit-learn 1.0.1

## Launch

To run the project locally, navigate to the app directory and install requirements.txt

    $ pip install -r requirements.txt

Run the application from terminal.

    $ python app.py

In order to get predicted values and check the model performance on the saved test set, run the following code.

    import csv
    import json
    import requests

    import pandas as pd

    from settings.constants import VAL_CSV

    # extract necessary columns and metrics according to specifications
    with open('settings/specifications.json') as f:
        specifications = json.load(f)

    info = specifications['description']
    x_columns, y_column, metrics = info['X'], info['y'], info['metrics']

    val_set = pd.read_csv(VAL_CSV, header=0)
    val_x, val_y = val_set[x_columns], val_set[y_column]

    # serialize data and send request
    req_data = {'data': json.dumps(val_x.to_dict())}
    response = requests.get('http://0.0.0.0:8000/predict', data=req_data)
    
    # get prediction and print the accuracy score
    api_predicted = response.json()['prediction']
    
    api_score = eval(metrics)(val_y, api_predicted)
    print('accuracy: ', api_score)

## To Do

In progress!
- Add a Jupyter notebook with EDA and model selection.
- Add Serializer class. Extend functionality and add more endpoints.
