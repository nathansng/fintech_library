---
title: "FinDL: DSC 180 Capstone Project"
header:
  overlay_image: /assets/images/finance_header.jpg
  caption: "Photo Credit: Unsplash"
  actions:
    - label: "Project GitHub Repo"
      url: "https://github.com/nathansng/fintech_library"
layout: single
classes: wide
author_profile: true
---

# FinDL: Deep Learning Library for Finance Applications

## Overview

Time series data is ubiquitous in today's world, particularly in the finance industry, where it is used for numerous tasks, such as forecasting. Developing effective deep learning models for time series forecasting requries extensive machine learning knowledge, which may be a barrier for financial specialists who lack this expertise. To address this challenge, we have developed FinDL, a library designed for both financial specialists and machine learning engineers.

FinDL provides an end-to-end machine learning pipeline for time series data, with out-of-the-box models that can be configured and fine-tuned according to the users' requirements. With this library, users can easily create and deploy machine learning models for finance-related tasks, such as future stock forecasting.

The library includes a data loader and data preprocessing functions, as well as time series forecasting models and loss visualization functions, to provide the tools necessary to build an end-to-end machine learning pipeline. The library has been developed in parallel with FinDL's NLP group, which focuses on building the tools for NLP applications in the finance industry.

<div style="text-align:center;">
        <img src="{{ site.url }}{{ site.baseurl }}/assets/images/FinDL_stack.png" alt="FinDL Module Stack">
    <p style="font-size: 15px">Stack visual of FinDL modules</p>
</div>

## Library Workflow

We have developed a comprehensive library that enables efficient processing of raw time series data into a machine learning model. Our library follows a well-defined pipeline that ensures ease of use and customization for each module.

To begin, we take the raw time series data and pass it through our data loader. The data loader is designed to filter and format data, making it suitable for futher processing. The output of the data loader is then fed into our data preprocesser, which employs advanced techniques, such as normalization and linear approximation to extract the trend and local feature information from the time series data.

The processed data is then ready to be used by our prediction model, which includes TreNet, LSTM, and CNN. Our model uses the processed data as input to generate predictions. Our model training executor takes charge of the training process and saves the best parameters of the model for the user.

Finally, users can utilize our visualization functions to produce compelling visualizations using the training and validation loss data. Our library provides an end-to-end solution that enables users to process, analyze, and visualize their time series data in a deep learning pipeline.

<div style="text-align:center;">
    <img style="width: 80%; height: auto;" src="{{ site.url }}{{ site.baseurl }}/assets/images/FinDL_workflow.png" alt="FinDL Workflow">
    <p style="font-size: 15px">FinDL workflow to create and train TreNet</p>
</div>


### Data Loader and Data Preprocessing

Our data loader is a versatile tool that can handle multiple file formats, including CSV and JSON. It's designed to simplify the process of quickly loading and processing large volumes of data. It also performs basic data filtering and formatting to ensure that only relevant data is included in the pipeline.

Once data is loaded in using FinDL's data loader, it can be seamlessly fed into our data preprocessing functions. These functions include normalization, which is used to rescale the data and ensure that it falls within a specific range. Additionally, FinDL includes advanced functions to extract trend slopes and durations from time series data, providing valuable insights into underlying trends and patterns. Another critical function the FinDL library provides creates samples of the time series data, which can then be separated into training, validation, and testing sets.

```
# Load and pre-process data
dl = DataLoader(**data_config)
la = LinearApproximation(dl.data, **la_hparams)
data = la.process_data()

# Normalize and linearly approximate data
trends, points = preprocessing.convert_data_points(data)
trend_scalar, points_scalaer = Scaler.Scaler(), Scaler.Scaler()
scaled_trends = trend_scaler.fit_transform(trends)
scaled_points = points_scaler.fit_transform(points)

trends, points = preprocessing.preprocess_data(scaled_trends, scaled_points, device, feat_config)
```


### Model Selection and Configuration

After the data has been processed, users can choose from a variety of models included in the FinDL library to start making time series forecasts. These models include popular options such as TreNet, LSTM, GRU, and CNN, and are ready to run out-of-the-box.

With these models, users can quickly and easily develop accurate and reliable time series predictions, even if they have limited machine learning experience. The models are highly customizable and can be fine-tuned to meet the specific needs of each user, ensuring that they deliver the best possible results for each unique dataset.

```
# Create model
model = TreNet.TreNet(device, LSTM_params=LSTM_hparams, CNN_params=CNN_hparams, **TreNet_hparams)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=training_config['lr'])
```


### Training the Model and Visualizations

FinDL makes model training and evaluation easy with its built-in training executor and visualization tool. The training executor takes care of training the chosen model using the provided data and saves the best performing model based on the lowest loss. The weights of the model are then saved in a file, making it easy for users to reproduce the model in the future.

Additionally, FinDL provides a visualization tool that generates graphs based on the loss values recorded during the training process. Users can easily pass the recorded losses into the visualization function to create a graph that displays the losses throughout the model's training. This helps users understand how the model performed during each epoch and provides valuable insights into how the model can be improved.

```
# Train model
train_loss, val_loss = train_models.train_loop(num_epochs, [X_train_trend, X_train_points], \
    y_train_trend, model, loss_fn, optimizer, \
    X_val=[X_val_trend, X_val_points], y_val=y_val_trend, printout=True, record_loss=True)

# Visualize loss during training
loss_visuals.visualize_loss([train_loss, val_loss], **visual_configs)
```


### Example Code API

The following code snippet demonstrates how to use FinDL to process raw time series data into trend data and then train a TreNet model:

```
# Load and pre-process data
dl = DataLoader(**data_config)
la = LinearApproximation(dl.data, **la_hparams)
data = la.process_data()

# Normalize and linearly approximate data
trends, points = preprocessing.convert_data_points(data)
trend_scalar, points_scalaer = Scaler.Scaler(), Scaler.Scaler()
scaled_trends = trend_scaler.fit_transform(trends)
scaled_points = points_scaler.fit_transform(points)

trends, points = preprocessing.preprocess_data(scaled_trends, scaled_points, device, feat_config)

# Create model
model = TreNet.TreNet(device, LSTM_params=LSTM_hparams, CNN_params=CNN_hparams, **TreNet_hparams)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=training_config['lr'])

# Train model
train_loss, val_loss = train_models.train_loop(num_epochs, [X_train_trend, X_train_points], \
    y_train_trend, model, loss_fn, optimizer, \
    X_val=[X_val_trend, X_val_points], y_val=y_val_trend, printout=True, record_loss=True)

# Visualize loss during training
loss_visuals.visualize_loss([train_loss, val_loss], **visual_configs)
```

This code illustrates how FinDL can be used to perform an end-to-end pipeline for time series forecasting, from data loading to model training and evaluation. By leveraging the library's powerful tools and out-of-the-box models, users can quickly and easily create accurate time series predictions for a wide range of finance-related tasks.