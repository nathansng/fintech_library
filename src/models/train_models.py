import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

"""
Create training loop
"""
def train_loop(n_epochs, X, y, model, loss_fn, optimizer, X_val=None, y_val=None, printout=False, record_loss=False):

    losses, val_losses = [], []
    for i in range(n_epochs):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        val_loss = None

        if X_val is not None and y_val is not None:
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val)

        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss per epoch
        if printout and i % 100 == 0:
            print(f"Epoch {i}:\n--------------")
            print(f"Train Loss: {np.sqrt(loss.item())}")
            if val_loss is not None:
                print(f"Validation Loss: {np.sqrt(val_loss.item())}")

            print()

        # Record loss per epoch
        if record_loss:
            losses.append(np.sqrt(loss.item()))
            if val_loss is not None:
                val_losses.append(np.sqrt(val_loss.item()))

    # Print final loss after training
    if printout:
        print(f"Final:\n--------------")
        print(f"Train Loss: {np.sqrt(loss.item())}")
        if val_loss is not None:
            print(f"Validation Loss: {np.sqrt(val_loss.item())}")

        print()

    if record_loss:
        return losses
