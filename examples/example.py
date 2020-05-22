# imports
import torch
import torch.nn    as nn
import torch.optim as optim
from   torchtrain import Module

class MyNetwork(Module):

    def __init__(self, size_input, size_hidden, size_output):
        """Create simple network"""
        # Initialise super
        super().__init__()

        # Set layers
        self.layer_1 = nn.Linear(size_input , size_hidden)
        self.layer_2 = nn.Linear(size_hidden, size_output)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        """Forward through network"""
        # Propagate layer 1
        out = self.layer_1(X)
        # Propagate layer 2
        out = self.layer_2(out)
        # Propagate softmax layer
        out = self.softmax(out)
        # Return result
        return out


if __name__ == "__main__":
    # Generate random data
    X_train = torch.rand((1024, 10))
    y_train = (torch.rand(1024)*10).to(torch.int64)
    X_test  = torch.rand((1024, 10))
    y_test  = (torch.rand(1024)*10).to(torch.int64)

    # Create network
    net = MyNetwork(10, 128, 10)

    # Fit network
    net.fit(X_train, y_train,
        epochs        = 10,
        batch_size    = 32,
        learning_rate = 0.01,
        criterion     = nn.NLLLoss(),
        optimizer     = optim.SGD,
        variable      = False,
        verbose       = True
    )

    # Predict network
    y_pred = net.predict(X_test,
        batch_size = 32,
        variable   = False,
        verbose    = True
    )
