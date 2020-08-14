import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader, TensorDataset
try:
    from torchtrain.variable_data_loader import VariableDataLoader
except:
    from variable_data_loader import VariableDataLoader

class Module(nn.Module):
    """Extention of nn.Module that adds fit and predict methods
        Can be used for automatic training.

        Attributes
        ----------
        progress : Progress()
            Used to track progress of fit and predict methods
    """

    def __init__(self, *args, **kwargs):
        """Only calls super method nn.Module with given arguments."""
        # Initialise super
        super().__init__(*args, **kwargs)

    def fit(self, X, y,
            epochs        = 10,
            batch_size    = 32,
            learning_rate = 0.01,
            criterion     = nn.NLLLoss(),
            optimizer     = optim.SGD,
            variable      = False,
            verbose       = True,
            **kwargs):
        """Train the module with given parameters

            Parameters
            ----------
            X : torch.Tensor
                Tensor to train with

            y : torch.Tensor
                Target tensor

            epochs : int, default=10
                Number of epochs to train with

            batch_size : int, default=32
                Default batch size to use for training

            learning_rate : float, default=0.01
                Learning rate to use for optimizer

            criterion : nn.Loss, default=nn.NLLLoss()
                Loss function to use

            optimizer : optim.Optimizer, default=optim.SGD
                Optimizer to use for training

            variable : boolean, default=False
                If True, accept inputs of variable length

            verbose : boolean, default=True
                If True, prints training progress

            Returns
            -------
            result : self
                Returns self
            """
        ################################################################
        #                Initialise training parameters                #
        ################################################################
        # Set optimiser
        optimizer = optimizer(
            params = self.parameters(),
            lr     = learning_rate
        )

        ################################################################
        #                         Prepare data                         #
        ################################################################

        # If the input length can be variable
        if variable:
            # Set device automatically
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Load data as variable length dataset
            data = VariableDataLoader(X, y, batch_size=batch_size, shuffle=True)

        # In the normal case
        else:
            # Get device
            device = X.device
            # Load data
            data = DataLoader(
                TensorDataset(X, y),
                batch_size = batch_size,
                shuffle    = True
            )

        ################################################################
        #                       Perform training                       #
        ################################################################

        # Loop over each epoch
        for epoch in range(1, epochs+1):
            try:
                # Loop over entire dataset
                for X_, y_ in tqdm.tqdm(data,
                    desc="[Epoch {:{width}}/{:{width}}]".format(
                        epoch, epochs, width=len(str(epochs)))):

                    # Clear optimizer
                    optimizer.zero_grad()

                    # Forward pass
                    # Get new input batch
                    X_ = X_.clone().detach().to(device)
                    # Run through module
                    y_pred = self(X_)
                    # Compute loss
                    loss = criterion(y_pred, y_)

                    # Backward pass
                    # Propagate loss
                    loss.backward()
                    # Perform optimizer step
                    optimizer.step()

            except KeyboardInterrupt:
                print("\nTraining interrupted, performing clean stop")
                break

        ################################################################
        #                         Returns self                         #
        ################################################################

        # Return self
        return self


    def predict(self, X, batch_size=32, variable=False, verbose=True, **kwargs):
        """Makes prediction based on input data X.
            Default implementation just uses the module forward(X) method,
            often the predict method will be overwritten to fit the specific
            needs of the module.

            Parameters
            ----------
            X : torch.Tensor
                Tensor from which to make prediction

            batch_size : int, default=32
                Batch size in which to predict items in X

            variable : boolean, default=False
                If True, accept inputs of variable length

            verbose : boolean, default=True
                If True, print progress of prediction

            Returns
            -------
            result : torch.Tensor
                Resulting prediction
            """
        # Do not perform gradient descent
        with torch.no_grad():
            # Initialise result
            result = list()
            indices = torch.arange(len(X))

            # If we expect variable input
            if variable:
                # Reset indices
                indices = list()

                # Load data
                data = VariableDataLoader(X, torch.zeros(len(X)),
                    index=True,
                    batch_size=batch_size,
                    shuffle=False
                )

                # Loop over data
                for X_, y_, i in tqdm.tqdm(data, desc="Predicting"):
                    # Perform prediction and append
                    result .append(self(X_))
                    # Store index
                    indices.append(i)

                # Concatenate inputs
                indices = torch.cat(indices)

            # If input is not variable
            else:
                # Predict each batch
                for batch in tqdm.tqdm(range(0, X.shape[0], batch_size),
                    desc="Predicting"):
                    
                    # Extract data to predict
                    X_ = X[batch:batch+batch_size]
                    # Add prediction
                    result.append(self(X_))

            # Concatenate result and return
            return torch.cat(result)[indices]


    def fit_predict(self, X, y,
            epochs        = 10,
            batch_size    = 32,
            learning_rate = 0.01,
            criterion     = nn.NLLLoss,
            optimizer     = optim.SGD,
            variable      = False,
            verbose       = True,
            **kwargs):
        """Train the module with given parameters

            Parameters
            ----------
            X : torch.Tensor
                Tensor to train with

            y : torch.Tensor
                Target tensor

            epochs : int, default=10
                Number of epochs to train with

            batch_size : int, default=32
                Default batch size to use for training

            learning_rate : float, default=0.01
                Learning rate to use for optimizer

            criterion : nn.Loss, default=nn.NLLLoss
                Loss function to use

            optimizer : optim.Optimizer, default=optim.SGD
                Optimizer to use for training

            variable : boolean, default=False
                If True, accept inputs of variable length

            verbose : boolean, default=True
                If True, prints training progress

            Returns
            -------
            result : torch.Tensor
                Resulting prediction
            """
        return self.fit(X, y,
                        epochs,
                        batch_size,
                        learning_rate,
                        criterion,
                        optimizer,
                        variable,
                        verbose,
                        **kwargs
            ).predict(X, batch_size, variable, verbose, **kwargs)
