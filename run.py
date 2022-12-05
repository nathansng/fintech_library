#!/usr/bin/env python

# Import libraries
import sys
import json
from torch import nn
from torch import optim

# Import files
from src.data import load_data
from src.features import preprocessing, Scaler
from src.models import TreNet, CNN, LSTM, train_models, setup

import pandas as pd

def main(targets):
    # Check device to store and run model on
    device = setup.find_device()

    if 'test' in targets:
        print("Running program on test data")
        targets += ['data', 'features', 'model']


    if 'data' in targets:
        # Load in data and separate into train, test sets
        with open('config/data_params.json') as f:
            data_cfg = json.load(f)
        processing = load_data.ProcessedData(**data_cfg['init'])
        processing.load_raw_data(**data_cfg['load'])
        data = processing.process_data()


    if 'features' in targets:
        with open('config/feature_params.json') as f:
            feature_cfg = json.load(f)

        data = pd.read_csv('./data/processed/processed_trends_10.csv')

        # Turn dataframe into tensors
        trends, points = preprocessing.convert_data_points(data)

        # Scale data
        trend_scaler = Scaler.Scaler()
        points_scaler = Scaler.Scaler()
        scaled_trends = trend_scaler.fit_transform(trends)
        scaled_points = points_scaler.fit_transform(points)

        # Create train, validation, and test sets
        (X_train_trend, y_train_trend, X_valid_trend, y_valid_trend, X_test_trend, y_test_trend), (X_train_points, X_valid_points, X_test_points) = preprocessing.preprocess_data(scaled_trends, scaled_points, device, feature_cfg)


    # if 'model' in targets:
    #     with open('config/model_params.json') as f:
    #         model_cfg = json.load(f)
    #     with open('config/training_params.json') as f:
    #         training_cfg = json.load(f)

    #     model = TreNet.TreNet(device=device, LSTM_params=model_cfg['LSTM'], CNN_params=model_cfg['CNN'], **model_cfg['TreNet'])
    #     loss_fn = nn.MSELoss()
    #     optimizer = optim.Adam(model.parameters(), lr=training_cfg['learning_rate'])

    #     training_loss = train_models.train_loop(training_cfg['num_epochs'], [X_train_trend, X_train_points], y_train_trend.reshape(y_train_trend.shape[0], 2), model, loss_fn, optimizer, printout=True, record_loss=True)


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)