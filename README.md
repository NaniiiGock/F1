# F1 ML Project Report

## Project Overview
This project aims to predict the winners of Formula 1 races based on historical data and explore other patterns such as pit stops and lap times improvements. 

The dataset containing race results, driver details, constructors, lap times, is sourced from:
- Kaggle (https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)
- FastF1 (https://docs.fastf1.dev/)


## Data Collection and Preprocessing

Data Preprocessing: The collected data needed cleaning, including:

- Handling missing values.
- Converting data types for consistency.
- Merging datasets if needed to create a comprehensive dataset.
- Feature engineering (creating additional features like race duration, driver experience, constructor performance, etc.).

  
## Exploratory Data Analysis

Visualizations: 


## Tested Models (different for different tasks): 

### Classification: For predicting race winners. 
- Logistic Regression
- Random Forest
- XGBoost are suitable.
  
### Regression: For predicting pit stop times or lap times. 

- Linear Regression
- Ridge/Lasso Regression
- Decision Trees
  
### Feature Selection: 
The most relevant features for training were selected based on the analysis of heatmap of relations of the parameters, their affect on each other

Cathegories:

- Driver-related attributes (experience, previous wins and places ...).
- Constructor-related attributes (general team performance).
- Track-related attributes (circuit type, length, weather conditions).

## Model Evaluation and Tuning

*__Metrics__*: accuracy, precision, recall, F1-score(for classification models),  MSE and MAE (for r egression models).

*__Hyperparameter Tuning__*:  grid search, random search and optimizers were used to find the hyperparameters for the selected models(B&W).

*__Model Interpretation__*: feature importance (shap values) were tested to understand which factors contribute most to predictions.

## Prediction and Analysis



## Conclusion

Alternative data sources for more robust training: 
Ensembles to improve accuracy.
Real-time prediction models for ongoing F1 seasons (transfer learning, tuning the previous model after each year, not to train the new one)


# Installation and usage:

Python Version: 3.11.7

pip install -r requirements.txt


# References:

