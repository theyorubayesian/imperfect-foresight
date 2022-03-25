# Next Event Prediction
This repository contains starter code for predicting whether the next event in an eCommerce application is a purchase or not.

## Setup
- Create your environment
```
make create-env
```
- Hack away. 🔨 🔨

## Data
This project uses the eCommerce Event History dataset available on [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-electronics-store).
You can download the dataset by running the following command
```
make get-data
```
You will need your Kaggle API in order for this command to work. 

## Experiments
Two models were investigated in the modelling notebook:
- Logistic Regression
- Decision Tree Classifier

No extensive hyperparameter tuning was performed on either model. Trial runs were logged to this Weights & Biases project: [Imperfect Foresight](https://wandb.ai/theyorubayesian/next-event-prediction/reports/Imperfect-Foresight--VmlldzoxNzQxMTgw)
