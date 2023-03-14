FinDL Tutorial
================

The FinDL provides all the tools needed to create an end-to-end machine learning pipeline. In this example, we demonstrate how to use FinDL's modules to create and run a TreNet model on time series data.

For a deeper look into how FinDL works, check out our website at `FinDL <https://nathansng.github.io/fintech_library/>`_.



Data Loader and Data Preprocessing
-----------------------------------

We first load and preprocess our data to prepare it for machine learning.

::

    # Load and pre-process data
    dl = DataLoader(data_config)
    la = LinearApproximation(dl.data, la_hparams)
    data = la.process_data()

    # Normalize and linearly approximate data
    trends, points = preprocessing.convert_data_points(data)
    data_scaler = Scaler.MultiScaler(2)
    scaled_trends, scaled_points = data_scaler.fit_transform([trends, points])

    split_data = preprocessing.preprocess_trends(scaled_trends, scaled_points, device, feature_cfg)


Model Selection and Configuration
------------------------------------

Then we create our models and select PyTorch optimizers and losses to use for our model training.

::

    # Create model
    model = TreNet.TreNet(device, LSTM_params=LSTM_hparams, CNN_params=CNN_hparams, TreNet_hparams)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config['lr'])

Training the Model and Visualizations
----------------------------------------

Finally, we train our model and plot the training and validation loss across our training epochs.

::

    # Train model
    trainer = train_models.TrainingExecutor(model=model, optim=optimizer, loss=loss_fn)
    trainer.train(split_data.get("X_train"), split_data.get("y_train"), X_val=split_data.get("X_valid"), y_val=split_data.get("y_valid"))
    train_loss, val_loss = trainer.losses["train"], trainer.losses["valid"]

    # Visualize loss during training
    loss_visuals.visualize_loss([train_loss, val_loss], visual_configs)


Full API Example
-----------------

Putting together our code, we're able to load and process time series data, create and train a model, and visualize the performance of our model during its training using FinDL.

::

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

