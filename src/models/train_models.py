import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

"""
Training Executor
"""
class TrainingExecutor:
    """ Training excutor that trains models using a given model, optimizer, and loss function. Uses a training loop and provided data to train models.

    Args:
        model (FinDL model): Deep learning model to train
        optim (PyTorch optimizer): PyTorch optimizer to use for training
        loss (PyTorch loss function): PyTorch loss function to use for training
        n_epochs (int, optional): Number of training epochs
        printout (bool, optional): If *True*, prints losses for every 100 epochs

    Returns:
        None
    """

    def __init__(self, model = None, optim=None, loss=None, n_epochs=2000, printout=True):
        self.model = model
        self.optimizer = optim
        self.loss_fn = loss
        self.n_epochs = n_epochs
        self.printout = printout
        self.losses = {"train": [], "valid": [], "test": []}


    def train(self, X, y, X_val=None, y_val=None, model_path="best_model"):
        """ Trains the model using the optimizer and loss functions on the provided data. Stores the parameters of the best performing model to reproduce models for later use.

        Args:
            X (tensor): Time series data
            y (tensor): Labels of time series data
            X_val (tensor, optional): Validation set of time series data
            y_val (tensor, optional): Validation set of corresponding labels
            model_path (str, optional): Path to store best model parameters

        Returns:
            None
        """

        best_model = None
        best_loss = float("inf")
        validation = False

        # Add data for validation loss
        if X_val is not None and y_val is not None:
            validation = True

        for i in range(self.n_epochs):
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Compute validation loss
            if validation:
                val_pred = self.model(X_val)
                val_loss = self.loss_fn(val_pred, y_val)

            # Store best performing model
            if validation:
                if val_loss.item() < best_loss:
                    best_loss = val_loss.item()
                    best_model = self.model.state_dict()
                    torch.save(self.model.state_dict(), model_path)
            else:
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_model = self.model.state_dict()
                    torch.save(self.model.state_dict(), model_path)


            # Perform backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Record loss per epoch
            self.losses["train"].append(np.sqrt(loss.item()))
            if val_loss is not None:
                self.losses["valid"].append(np.sqrt(val_loss.item()))

            # Print loss per epoch
            if self.printout and i % 100 == 0:
                if val_loss is not None:
                    self.print_loss(f"Epoch {i}", self.losses["train"][-1], self.losses["valid"][-1])
                else:
                    self.print_loss(f"Epoch {i}", self.losses["train"][-1])


        # Calculate final model losses
        pred = self.model(X)
        final_train_loss = self.loss_fn(pred, y)

        if validation:
            val_pred = self.model(X_val)
            final_val_loss = self.loss_fn(val_pred, y_val)

        # Update model to best performing model
        self.model.load_state_dict(torch.load(model_path))
        pred = self.model(X)
        best_train_loss = self.loss_fn(pred, y)

        if validation:
            val_pred = self.model(X_val)
            best_val_loss = self.loss_fn(val_pred, y_val)

        # Print final loss after training and best performing model
        if self.printout:
            if validation:
                self.print_loss("Final", np.sqrt(final_train_loss.item()), np.sqrt(final_val_loss.item()))
                self.print_loss("Best Model", np.sqrt(best_train_loss.item()), np.sqrt(best_val_loss.item()))
            else:
                self.print_loss("Final", np.sqrt(final_train_loss.item()))
                self.print_loss("Best Model", np.sqrt(best_train_loss.item()))


    def print_loss(self, label, train_loss, valid_loss=None):
        """ Template for printing losses, prints out losses

        Args:
            label (str): Label/context of printed loss
            train_loss (int or float): Training loss to print
            valid_loss (int or float, optional): Validation loss to print

        Returns:
            None
        """

        print(f"{label}:\n--------------")
        print(f"Train Loss: {train_loss}")

        if valid_loss is not None:
            print(f"Validation Loss: {valid_loss}")
        print()


    def clear_recorded_loss(self):
        """ Resets the saved losses, useful when training new models

        Returns:
            None
        """

        self.losses = {"train": [], "valid": [], "test": []}