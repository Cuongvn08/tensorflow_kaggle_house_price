# A tensorflow based standard NN to explore data and linear regression for Kaggle house price competition

This project presents how to use Tensorflow to implement a standard NN for Kaggle
house price competition.
+ config.py: consists of all settings, hyper-parameters
+ data.py: to load data from csv and separate into data and label
    + train.csv, test.csv: were downloaded from kaggle homepage, mnist project
+ model.py: implement Model class to design expected model, get logit and 
weights or biases
+ train.py: run this file to train and evaluate the model after setting, 
automatically save logs and checkpoint into result folder
+ run.py: set the best step with the lowest cost and run this file to get
prediction of test data and submit.

Development enviroment:
+ Window 10
+ Anaconda 3
+ Python 3.5
+ Tensorflow 1.1
+ Pycharm community
