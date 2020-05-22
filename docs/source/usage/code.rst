Code
====
To use torch-train into your own project, you can use it in place of the `torch.nn.Module`_.
Here we show some simple examples on how to use the torch-train Module in your own python code.
For a complete documentation we refer to the :ref:`Reference` guide.

.. _`torch.nn.Module`: https://pytorch.org/docs/stable/nn.html#module

Import
^^^^^^
To import the Module use

.. code:: python

  from torchtrain import Module

Working example
^^^^^^^^^^^^^^^

In this example, we create a basic torch Module and use its :py:meth:`fit` and :py:meth:`predict` methods to train and test.
First we import ``torch`` and the ``torchtrain`` Module

.. code:: python

  # imports
  import torch
  import torch.nn as nn
  from   torchtrain import Module

Next we create our simple network consisting of 2 layers and a softmax output function.

.. note::
  We extend the ``torchtrain.Module`` instead of the ``torch.nn.Module`` like you normally would.

Furthermore we implement the :py:meth:`forward()` method to propagate through the network.

.. code:: python

  class MyNetwork(Module):

      def __init__(self, size_input, size_hidden, size_output):
          """Create simple network"""
          # Initialise super
          super().__init__()

          # Set layers
          self.layer_1 = nn.Linear(size_input , size_hidden)
          self.layer_2 = nn.Linear(size_hidden, size_output)
          self.softmax = nn.LogSoftmax(dim=1)

      def forward(X):
          """Forward through network"""
          # Propagate layer 1
          out = self.layer_1(X)
          # Propagate layer 2
          out = self.layer_2(out)
          # Propagate softmax layer
          out = self.softmax(out)
          # Return result
          return out

Now that we have created our network, we generate some random training and testing data.

.. code:: python

    # Generate random data
    X_train = torch.rand((1024, 10))
    y_train = (torch.rand(1024)*10).to(torch.int64)
    X_test  = torch.rand((1024, 10))
    y_test  = (torch.rand(1024)*10).to(torch.int64)

Finally, we create the network and invoke its :py:meth:`fit` and :py:meth:`predict` methods.

.. code:: python

  # Create network
  net = MyNetwork(10, 128, 10)

  # Fit network
  net.fit(X_train, y_train,
      epochs        = 10,
      batch_size    = 32,
      learning_rate = 0.01,
      criterion     = nn.NLLLoss,
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
