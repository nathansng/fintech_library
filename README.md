# FinDL - Financial Deep Learning Library

## DSC 180 Project

The goal of this project is to create a deep learning and machine learning library that allows users to easily create and deploy machine learning models for finance related tasks, such as future stock forecasting. This repo contains a data loader, data preprocessing functions, time series forecasting models, and loss visualization functions to provide an end-to-end machine learning and visualization pipeline. This project is made in parallel with the finance library's NLP group. 

### Downloading Data

Our data is downloaded from Kaggle.com. The dataset that we used for our experiment is the Stock Exchange Data created by Cody in 2018. The dataset is available at: https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data. The dataset we used from this Kaggle dataset is "indexProcessed.csv". Save the csv file in the path `./data/raw/`. 

### Models 

The following models have been implemented in the current implementation of the library. 

- CNN
- LSTM
- TreNet

The Convolutional Neural Network (CNN) takes raw data points as input, extracts and learns the local feature information, and outputs the predicted local feature. The tuneable parameters of CNN are as follows:

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

*Note*: Running the code on a gpu will make the program run significantly faster than only cpu. The code also benefits with more RAM, as too low of memory will kill the process. 

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

You can also specify the model to run by specifying the model name in addition to any targets listed above by using `python run.py [targets] [model]`. The default model used if no model is specified is TreNet. The available models and their description are listed below: 

- `trenet`: Runs the TreNet model. Processes 1 dimensional time series data into a sequence of linear regressions encoded by the regressions slope and duration. Uses trend slope and duration data to predict future trends. 

- `lstm`: Runs an LSTM model. Trains on 1 dimensional time series data to predict future time series data. 

- `cnn`: Runs the CNN stack from the TreNet model. Trains on 1 dimensional time series data to predict future time series data. 


