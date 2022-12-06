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
from src.visualization import loss_visuals

def main(targets):
    # Check device to store and run model on
    device = setup.find_device()
       
    if 'all' in targets: 
        print("Running program on data\n")
        targets += ['data', 'features', 'model', 'visual']
        
        # Load in actual data configurations
        with open('config/data_params.json') as f:
            data_cfg = json.load(f)
        

    if 'test' in targets:
        print("Running program on test data\n")
        targets += ['data', 'features', 'model', 'visual']
        
        # Load in test data configurations
        with open('config/test_data_params.json') as f:
            data_cfg = json.load(f)


    if 'data' in targets:
        print("Running data target\n")
        print("Loading in trends data\n") 
        
        # Load in data and separate into train, test sets
        processing = load_data.ProcessedData(**data_cfg['init'])
        processing.load_raw_data(**data_cfg['load'])
        data = processing.process_data()


    if 'features' in targets:
        print("Running features target\n")
        print("Preprocessing data...\n")
        
        with open('config/feature_params.json') as f:
            feature_cfg = json.load(f)

        # Turn dataframe into tensors
        trends, points = preprocessing.convert_data_points(data)

        # Scale data
        trend_scaler = Scaler.Scaler()
        points_scaler = Scaler.Scaler()
        scaled_trends = trend_scaler.fit_transform(trends)
        scaled_points = points_scaler.fit_transform(points)

        # Create train, validation, and test sets
        (X_train_trend, y_train_trend, X_valid_trend, y_valid_trend, X_test_trend, y_test_trend), (X_train_points, X_valid_points, X_test_points) = preprocessing.preprocess_data(scaled_trends, scaled_points, device, feature_cfg)


    if 'model' in targets:
        print("Running model target\n")
        print("Training model...\n")
        
        with open('config/model_params.json') as f:
            model_cfg = json.load(f)
        with open('config/training_params.json') as f:
            training_cfg = json.load(f)

        # Initialize model, loss, optimizer
        model = TreNet.TreNet(device=device, LSTM_params=model_cfg['LSTM'], CNN_params=model_cfg['CNN'], **model_cfg['TreNet']).to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=training_cfg['learning_rate'])

        # Train model
        training_loss = train_models.train_loop(training_cfg['num_epochs'], [X_train_trend, X_train_points], y_train_trend.reshape(y_train_trend.shape[0], 2), model, loss_fn, optimizer, printout=True, record_loss=True)
        
        
    if 'visual' in targets: 
        print("Running visual target\n")
        print("Creating loss visualization...\n")
        
        with open('config/visual_params.json') as f:
            visual_cfg = json.load(f)
        
        # Visualize loss per epoch and save plot 
        loss_visuals.create_dir(visual_cfg['path'])
        loss_visuals.visualize_loss(training_loss, **visual_cfg)


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)