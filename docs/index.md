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

---

## Overview

Time series data is ubiquitous in today's world, particularly in the finance industry, where it is used for numerous tasks, such as forecasting. Developing effective deep learning models for time series forecasting requries extensive machine learning knowledge, which may be a barrier for financial specialists who lack this expertise. To address this challenge, we have developed FinDL, a library designed for both financial specialists and machine learning engineers.

FinDL provides an end-to-end machine learning pipeline for time series data, with out-of-the-box models that can be configured and fine-tuned according to the users' requirements. With this library, users can easily create and deploy machine learning models for finance-related tasks, such as future stock forecasting.

The library includes a data loader and data preprocessing functions, as well as time series forecasting models and loss visualization functions, to provide the tools necessary to build an end-to-end machine learning pipeline. The library has been developed in parallel with FinDL's NLP group, which focuses on building the tools for NLP applications in the finance industry.

<div style="text-align:center;">
    <img src="/assets/images/FinDL_stack.png" alt="FinDL Module Stack">
    <p style="font-size: 15px">Stack visual of FinDL modules</p>
</div>

## Library Workflow

We have developed a comprehensive library that enables efficient processing of raw time series data into a machine learning model. Our library follows a well-defined pipeline that ensures ease of use and customization for each module.

To begin, we take the raw time series data and pass it through our data loader. The data loader is designed to filter and format data, making it suitable for futher processing. The output of the data loader is then fed into our data preprocesser, which employs advanced techniques, such as normalization and linear approximation to extract the trend and local feature information from the time series data.

The processed data is then ready to be used by our prediction model, which includes TreNet, LSTM, and CNN. Our model uses the processed data as input to generate predictions. Our model training executor takes charge of the training process and saves the best parameters of the model for the user.

Finally, users can utilize our visualization functions to produce compelling visualizations using the training and validation loss data. Our library provides an end-to-end solution that enables users to process, analyze, and visualize their time series data in a deep learning pipeline.

<div style="text-align:center;">
    <img style="width: 80%; height: auto;" src="/assets/images/FinDL_workflow.png" alt="FinDL Workflow">
    <p style="font-size: 15px">FinDL workflow to create and train TreNet</p>
</div>

