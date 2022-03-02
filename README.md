![](https://forthebadge.com/images/badges/made-with-python.svg)

# Disaster-Response-Project
Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages

-----------
### Table of Contents

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing and Acknowledgements

* [![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)
* This project is part of [Udacity](https://www.udacity.com/) Data Scientist Nanodegree. 
* Credit must be given to [Figure Eight](https://www.figure-eight.com/?ref=Welcome.AI) for the data. 


**Copyright (c) 2021 Juma Hamdan**