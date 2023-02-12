import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

"""
Create training loop
"""
def train_loop(n_epochs, X, y, model, loss_fn, optimizer, X_val=None, y_val=None, printout=False, record_loss=False, model_path="best_model"):
    train_losses = []
    best_model = None
    best_loss = float("inf")
    validation = False
    
    # Add data for validation loss
    if X_val is not None and y_val is not None: 
        validation = True
        val_losses = []
    
    for i in range(n_epochs):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Compute validation loss 
        if validation:
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val)
            
        # Store best performing model
        if validation: 
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                best_model = model.state_dict()
                torch.save(model.state_dict(), model_path)
        else:
            if loss.item() < best_loss: 
                best_loss = loss.item()
                best_model = model.state_dict()
                torch.save(model.state_dict(), model_path)
                

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
            train_losses.append(np.sqrt(loss.item()))
            if validation:
                val_losses.append(np.sqrt(val_loss.item()))
    
    # Calculate final model losses 
    pred = model(X)
    final_train_loss = loss_fn(pred, y)
    
    if validation:
        val_pred = model(X_val)
        final_val_loss = loss_fn(val_pred, y_val)
    
    # Update model to best performing model
    model.load_state_dict(torch.load(model_path))
    pred = model(X)
    best_train_loss = loss_fn(pred, y)
    
    if validation:
        val_pred = model(X_val)
        best_val_loss = loss_fn(val_pred, y_val)

    # Print final loss after training and best performing model 
    if printout:
        print(f"Final:\n--------------")
        print(f"Train Loss: {np.sqrt(final_train_loss.item())}")
        if validation:
            print(f"Validation Loss: {np.sqrt(final_val_loss.item())}")
        print()
        
        print(f"Best Model:\n--------------")
        print(f"Train Loss: {np.sqrt(best_train_loss.item())}")
        if validation:
            print(f"Validation Loss: {np.sqrt(best_val_loss.item())}")
        print()

    if record_loss: 
        if validation:
            return train_losses, val_losses
        else:
            return train_losses