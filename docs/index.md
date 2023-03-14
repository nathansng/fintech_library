---
title: "FinDL: DSC 180 Capstone Project"
header:
  overlay_image: /assets/images/finance_header.jpg
  caption: "Photo Credit: Unsplash"
  actions:
    - label: "Project GitHub Repo"
      url: "https://github.com/nathansng/fintech_library"
    - label: "FinDL Documentation"
      url: "https://fintech-library.readthedocs.io/en/latest/"
gallery:
    - url: /assets/images/FinDL_Stack2.png
      image_path: assets/images/FinDL_Stack2.png
      alt: "FinDL Module Stack"
      title: "FinDL Module Stack"
gallery2:
    - url: /assets/images/FinDL_Workflow2.png
      image_path: assets/images/FinDL_Workflow2.png
      alt: "FinDL Workflow"
      title: "FinDL Workflow"
gallery3:
    - url: /assets/images/loss_trenet.png
      image_path: /assets/images/loss_trenet.png
      alt: "Trenet losses"
      title: "TreNet Training and Validation Loss"
    - url: /assets/images/loss_lstm.png
      image_path: /assets/images/loss_lstm.png
      alt: "LSTM losses"
      title: "LSTM Training and Validation Loss"
    - url: /assets/images/loss_cnn.png
      image_path: /assets/images/loss_cnn.png
      alt: "CNN losses"
      title: "CNN Training and Validation Loss"
layout: single
classes: wide
author_profile: true
---

# FinDL: Deep Learning Library for Finance Applications

## Overview

Time series data is ubiquitous in today's world, particularly in the finance industry, where it is used for numerous tasks, such as forecasting. Developing effective deep learning models for time series forecasting requries extensive machine learning knowledge, which may be a barrier for financial specialists who lack this expertise. To address this challenge, we have developed FinDL, a library designed for both financial specialists and machine learning engineers.

FinDL provides an end-to-end machine learning pipeline for time series data, with out-of-the-box models that can be configured and fine-tuned according to the users' requirements. With this library, users can easily create and deploy machine learning models for finance-related tasks, such as future stock forecasting.

The library includes a data loader and data preprocessing functions, as well as time series forecasting models and loss visualization functions, to provide the tools necessary to build an end-to-end machine learning pipeline. The library has been developed in parallel with FinDL's NLP group, which focuses on building the tools for NLP applications in the finance industry.

{% include gallery id="gallery" caption="Stack visual of FinDL modules" %}

## Library Workflow

We have developed a comprehensive library that enables efficient processing of raw time series data into a machine learning model. Our library follows a well-defined pipeline that ensures ease of use and customization for each module.

To begin, we take the raw time series data and pass it through our data loader. The data loader is designed to filter and format data, making it suitable for futher processing. The output of the data loader is then fed into our data preprocesser, which employs advanced techniques, such as normalization and linear approximation to extract the trend and local feature information from the time series data.

The processed data is then ready to be used by our prediction model, which includes TreNet, LSTM, and CNN. Our model uses the processed data as input to generate predictions. Our model training executor takes charge of the training process and saves the best parameters of the model for the user.

Finally, users can utilize our visualization functions to produce compelling visualizations using the training and validation loss data. Our library provides an end-to-end solution that enables users to process, analyze, and visualize their time series data in a deep learning pipeline.

{% include gallery id="gallery2" caption="FinDL workflow to create and train TreNet" %}


### Data Loader and Data Preprocessing

Our data loader is a versatile tool that can handle multiple file formats, including CSV and JSON. It's designed to simplify the process of quickly loading and processing large volumes of data. It also performs basic data filtering and formatting to ensure that only relevant data is included in the pipeline.

Once data is loaded in using FinDL's data loader, it can be seamlessly fed into our data preprocessing functions. These functions include normalization, which is used to rescale the data and ensure that it falls within a specific range. Additionally, FinDL includes advanced functions to extract trend slopes and durations from time series data, providing valuable insights into underlying trends and patterns. Another critical function the FinDL library provides creates samples of the time series data, which can then be separated into training, validation, and testing sets.

```
# Load and pre-process data
dl = DataLoader(data_config)
la = LinearApproximation(dl.data, la_hparams)
data = la.process_data()

# Normalize and linearly approximate data
trends, points = preprocessing.convert_data_points(data)
data_scaler = Scaler.MultiScaler(2)
scaled_trends, scaled_points = data_scaler.fit_transform([trends, points])

split_data = preprocessing.preprocess_trends(scaled_trends, scaled_points, device, feature_cfg)
```


### Model Selection and Configuration

After the data has been processed, users can choose from a variety of models included in the FinDL library to start making time series forecasts. These models include popular options such as TreNet, LSTM, GRU, and CNN, and are ready to run out-of-the-box.

With these models, users can quickly and easily develop accurate and reliable time series predictions, even if they have limited machine learning experience. The models are highly customizable and can be fine-tuned to meet the specific needs of each user, ensuring that they deliver the best possible results for each unique dataset.

```
# Create model
model = TreNet.TreNet(device, LSTM_params=LSTM_hparams, CNN_params=CNN_hparams, TreNet_hparams)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=training_config['lr'])
```


### Training the Model and Visualizations

FinDL makes model training and evaluation easy with its built-in training executor and visualization tool. The training executor takes care of training the chosen model using the provided data and saves the best performing model based on the lowest loss. The weights of the model are then saved in a file, making it easy for users to reproduce the model in the future.

Additionally, FinDL provides a visualization tool that generates graphs based on the loss values recorded during the training process. Users can easily pass the recorded losses into the visualization function to create a graph that displays the losses throughout the model's training. This helps users understand how the model performed during each epoch and provides valuable insights into how the model can be improved.

```
# Train model
trainer = train_models.TrainingExecutor(model=model, optim=optimizer, loss=loss_fn)
trainer.train(split_data.get("X_train"), split_data.get("y_train"), X_val=split_data.get("X_valid"), y_val=split_data.get("y_valid"))
train_loss, val_loss = trainer.losses["train"], trainer.losses["valid"]

# Visualize loss during training
loss_visuals.visualize_loss([train_loss, val_loss], visual_configs)
```


### Example Code API

The following code snippet demonstrates how to use FinDL to process raw time series data into trend data and then train a TreNet model:

```
# Load and pre-process data
dl = DataLoader(data_config)
la = LinearApproximation(dl.data, la_hparams)
data = la.process_data()

# Normalize and linearly approximate data
trends, points = preprocessing.convert_data_points(data)
data_scaler = Scaler.MultiScaler(2)
scaled_trends, scaled_points = data_scaler.fit_transform([trends, points])

split_data = preprocessing.preprocess_trends(scaled_trends, scaled_points, device, feature_cfg)

# Create model
model = TreNet.TreNet(device, LSTM_params=LSTM_hparams, CNN_params=CNN_hparams, TreNet_hparams)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=training_config['lr'])

# Train model
trainer = train_models.TrainingExecutor(model=model, optim=optimizer, loss=loss_fn)
trainer.train(split_data.get("X_train"), split_data.get("y_train"), X_val=split_data.get("X_valid"), y_val=split_data.get("y_valid"))
train_loss, val_loss = trainer.losses["train"], trainer.losses["valid"]

# Visualize loss during training
loss_visuals.visualize_loss([train_loss, val_loss], visual_configs)
```

This code illustrates how FinDL can be used to perform an end-to-end pipeline for time series forecasting, from data loading to model training and evaluation. By leveraging the library's powerful tools and out-of-the-box models, users can quickly and easily create accurate time series predictions for a wide range of finance-related tasks.



### Results

To demonstrate the use of FinDL, we also implemented deep learning pipelines using LSTM and CNN to predict the next stock price in addition to the trend prediction task TreNet model we created. Using FinDL's components to create and run our pipelines, we generated plots of the training performance of the models when we ran the models through our training executor. We were able to build and run three different model pipelines with different tasks using FinDL components. Our library design allowed us to easily and efficiently swap components out to generate these three models and visuals.

{% include gallery id="gallery3" caption="Training and Validation Loss of FinDL Models" %}



### Conclusion

In this project, we created FinDL, a library that provides users the tools to create an end-to-end machine learning pipeline for time series tasks in the finance domain. With FinDL's modularized components, we simplified the process of data processing, model selection, and model training to allow users of all experiences to build effective time series deep learning models. While less experienced users can easily build good performing base line models without much background knowledge in deep learning, more experienced users can use the wide variety of customization and configuration options FinDL provides to optimize and improve their time series models. In the future, we will continue building upon the library that we have and implement more models and introduce additional features to cover a wider range of deep learning prediction tasks in the finance domain.
