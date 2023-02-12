#!/usr/bin/env python

# Import libraries
import sys
import json
from torch import nn
from torch import optim

# Import files
from src.data.data_loader import DataLoader
from src.data.linear_approximation import LinearApproximation
from src.features import preprocessing, Scaler
from src.models import TreNet, CNN, LSTM, train_models, setup
from src.visualization import loss_visuals

def main(targets):
    # Check device to store and run model on
    device = setup.find_device()
    test = False
    
    # Check if model specified 
    if 'lstm' not in targets and 'cnn' not in targets and 'trenet' not in targets:
        print("Specify model to run: lstm, cnn, trenet")
        print("Running Default Model: TreNet")
        targets += ['trenet']

    # Add targets based on all or test
    if 'all' in targets:
        print("Running program on data\n")
        targets += ['data', 'features', 'model', 'visual']

    if 'test' in targets:
        print("Running program on test data\n")
        targets += ['data', 'features', 'model', 'visual']
        test = True
        
    # Read configurations 
    if 'data' in targets: 
        if test:
            # Load in test data configurations
            with open('config/test_data_params.json') as f:
                data_cfg = json.load(f)
        else:
            # Load in actual data configurations
            with open('config/data_params.json') as f:
                data_cfg = json.load(f)
                
    if 'features' in targets: 
        with open('config/feature_params.json') as f:
            feature_cfg = json.load(f)
            
    if 'model' in targets: 
        with open('config/model_params.json') as f:
            model_cfg = json.load(f)
        with open('config/training_params.json') as f:
            training_cfg = json.load(f)
            
    if 'visual' in targets: 
        with open('config/visual_params.json') as f:
            visual_cfg = json.load(f)

    
    # Run TreNet model 
    if 'trenet' in targets: 
        if 'data' in targets:
            print("Running data target\n")
            print("Loading in trends data\n")

            # Load data and apply linear approximation
            dl = DataLoader(**data_cfg['load'])
            la = LinearApproximation(data=dl.data, **data_cfg['linear_approximation'])
            data = la.process_data()


        if 'features' in targets:
            print("Running features target\n")
            print("Preprocessing data...\n")

            # Turn dataframe into tensors
            trends, points = preprocessing.convert_data_points(data)

            # Scale data
            trend_scaler = Scaler.Scaler()
            points_scaler = Scaler.Scaler()
            scaled_trends = trend_scaler.fit_transform(trends)
            scaled_points = points_scaler.fit_transform(points)

            # Create train, validation, and test sets
            (X_train_trend, y_train_trend, X_val_trend, y_val_trend, X_test_trend, y_test_trend), \
            (X_train_points, X_val_points, X_test_points) = preprocessing.preprocess_data(scaled_trends, scaled_points, device, feature_cfg)


        if 'model' in targets:
            print("Running model target\n")
            print("Training model...\n")

            # Initialize model, loss, optimizer
            model = TreNet.TreNet(device=device, LSTM_params=model_cfg['TreNet']['LSTM'], CNN_params=model_cfg['TreNet']['CNN'], **model_cfg['TreNet']['TreNet']).to(device)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=training_cfg['learning_rate'])

            # Train model
            train_loss, val_loss = train_models.train_loop(training_cfg['num_epochs'], [X_train_trend, X_train_points], \
                                                    y_train_trend.reshape(y_train_trend.shape[0], 2), model, loss_fn, optimizer,\
                                                    X_val=[X_val_trend, X_val_points], y_val=y_val_trend.reshape(y_val_trend.shape[0], 2), \
                                                    printout=True, record_loss=True)
            
    # Run LSTM model
    if 'lstm' in targets: 
        if 'data' in targets:
            print("Running data target\n")
            print("Loading in time series data\n")

            # Load data and apply linear approximation
            dl = DataLoader(**data_cfg['load'])
            data = dl.data.loc[:, "Close"].reset_index(drop=True)


        if 'features' in targets:
            print("Running features target\n")
            print("Preprocessing data...\n")
            
            # Normalize data
            scaler = Scaler.Scaler()
            data = scaler.fit_transform(data).to(device)
            
            # Create batches to train on
            X, y = preprocessing.extract_data(data, **feature_cfg)
            
            # Create train, validation, and test sets
            X_train, y_train, X_val, y_val, X_test, y_test = preprocessing.train_valid_test_split(X, y, device=device)


        if 'model' in targets:
            print("Running model target\n")
            print("Training model...\n")

            # Initialize model, loss, optimizer
            model = LSTM.LSTM(device=device, **model_cfg['LSTM']).to(device)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=training_cfg['learning_rate'])

            # Train model
            train_loss, val_loss = train_models.train_loop(training_cfg['num_epochs'], X_train, y_train, model, loss_fn, optimizer, X_val=X_val, y_val=y_val, printout=True, record_loss=True)
            
    # Run CNN model
    if 'cnn' in targets: 
        if 'data' in targets:
            print("Running data target\n")
            print("Loading in time series data\n")

            # Load data and apply linear approximation
            dl = DataLoader(**data_cfg['load'])
            data = dl.data.loc[:, "Close"].reset_index(drop=True)


        if 'features' in targets:
            print("Running features target\n")
            print("Preprocessing data...\n")
            
            # Normalize data
            scaler = Scaler.Scaler()
            data = scaler.fit_transform(data).to(device)
            
            # Create batches to train on
            X, y = preprocessing.extract_data(data, **feature_cfg)
            
            # Create train, validation, and test sets
            X_train, y_train, X_val, y_val, X_test, y_test = preprocessing.train_valid_test_split(X, y, device=device)


        if 'model' in targets:
            print("Running model target\n")
            print("Training model...\n")

            # Initialize model, loss, optimizer
            model = CNN.TreNetCNN(device=device, **model_cfg['CNN']).to(device)
            loss_fn = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=training_cfg['learning_rate'])

            # Train model
            train_loss, val_loss = train_models.train_loop(training_cfg['num_epochs'], X_train, y_train, model, loss_fn, optimizer, X_val=X_val, y_val=y_val, printout=True, record_loss=True)


    # Visualize loss of training model
    if 'visual' in targets:
        print("Running visual target\n")
        print("Creating loss visualization...\n")

        with open('config/visual_params.json') as f:
            visual_cfg = json.load(f)

        # Visualize loss per epoch and save plot
        loss_visuals.create_dir(visual_cfg['path'])
        loss_visuals.visualize_loss([train_loss, val_loss], **visual_cfg)


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
