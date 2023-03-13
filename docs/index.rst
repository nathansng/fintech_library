.. FinDL documentation master file, created by
   sphinx-quickstart on Mon Mar 13 10:54:26 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FinDL's documentation!
=================================

FinDL is a deep learning and machine learning library that allows users to easily create and deploy machine learning models for finance related tasks. The library includes components to build an end-to-end machine learning pipeline, such as a data loader, data preprocessing functions, time series forecasting models, training executor, and loss visualization functions.


Getting Started:
-----------------

:doc:`code/overview`
    Overview of the FinDL Library and project.

:doc:`code/tutorials`
    Example of creating an end-to-end machine learning pipeline with FinDL.



API Documentation:
-------------------

:doc:`code/data`
    :ref:`data-loader`

:doc:`code/features`
    :ref:`preprocessing`

    :ref:`linear-approximation`

    :ref:`scaler`

:doc:`code/models`
    :ref:`trenet`

    :ref:`lstm`

    :ref:`gru`

    :ref:`cnn`

:doc:`code/model_training`
    :ref:`setup`

    :ref:`training_executor`

:doc:`code/visualization`
    :ref:`loss-visuals`


.. toctree::
   :maxdepth: 1
   :caption: Getting Started:
   :hidden:

   code/overview
   code/tutorials


.. toctree::
   :maxdepth: 2
   :caption: API Documentation:
   :hidden:

   code/data
   code/features
   code/models
   code/model_training
   code/visualization