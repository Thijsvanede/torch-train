Overview
========
This section explains the design of torch-train on a high level.
torch-train is a network that is implemented as an extension of :py:class:`torch.nn.Module`.
This means it can be trained and used as any neural network module in the pytorch library.

In addition, we provide automatic methods to train and predict given data tensors using our :ref:`Module` extension.
This follows a ``scikit-learn`` approach with :py:meth:`fit`, :py:meth:`predict` and :py:meth:`fit_predict` methods.
We refer to the :ref:`Reference` for a detailed description.
