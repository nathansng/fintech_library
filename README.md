# FinDL - Financial Deep Learning Library

## DSC 180A Quarter 1 Project 

The goal of this project is to create a deep learning and machine learning library that allows users to easily create and deploy machine learning models for finance related tasks. 
This repo focuses on the initial implementations of time series forecasting model and will eventually be built upon in quarter 2 and combined with models made by the NLP group. 

### Downloading Data

Our data is downloaded from Kaggle.com. The dataset that we used for our experiment is the Stock Exchange Data created by Cody in 2018. The dataset is available at: https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data. The dataset we used from this Kaggle dataset is "indexProcessed.csv". Save the csv file in the path `./data/raw/`. 

### Models 

The following models have been implemented in the current implementation of the library. 

- CNN
- LSTM
- TreNet

The Convolutional Neural Net work (CNN) takes raw data points as input, extracts and learns the local feature information, and outputs the predicted local feature. The tuneable parameters of CNN are as follows:

- number of layers
- convolutional layer size 
- filter size
- dropout
- output size
- learning rate 

The Long Short Term Memory takes the trends' slope and duration, which are extracted by using linear approximation approach on the raw data points, learns the trend dependencies, and outputs the predicted slope and duration of the next trend. The tunable parameters of LSTM are as follows:

- input size
- hidden size
- number of layers
- output size
- learning rate.

TreNet takes the predicted results from both CNN and LSTM, and combines them using a fully connected layer to generate the predicted output. The tunable parameters of TreNet are as follows: 

- Hyperparameters of LSTM 
- Hyperparameters of CNN
- size of feature fusion layer
- output size 


### Running Code

*Note*: Running the code on a gpu will make the program run significantly faster than only cpu

To run the code, run `python run.py [target]` to run the corresponding target. The available targets and their description are listed below: 

- `all`: Runs all targets using actual data, data path can be specified in `./config/data_params.json` file

- `test`: Runs all targets using test data, test data can be found in `./test/testdata/testdata.csv`, test path can be specified in `./config/test_data_params.json` file

- `data`: Runs the data and feature loading code, which opens a dataset and converts the dataset into trend durations and slopes
  - Configure parameters in `./config/data_params.json`

- `features`: Runs the preprocessing code for the trends and stock data, normalizes all data, and splits data into training and testing sets for machine learning model use 
  - Configure parameters in `./config/feature_params.json`

- `model`: Runs the machine learning model training code 
  - Configure model parameters in `./config/model_params.json`
  - Configure training parameters in `./config/training_params.json`

- `visual`: Runs the loss plot visualization code which stores a line plot of the loss per epoch to a path 
  - Configure parameters in `./config/visual_params.json`




