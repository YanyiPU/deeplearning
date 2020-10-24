
Keras 评价指标
===============

1.常用评价指标
----------------------

1.1 Accuracy metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - Accuracy class

    - BinaryAccuracy class

    - CategoricalAccuracy class

    - TopKCategoricalAccuracy class

    - SparseTopKCategoricalAccuracy class


1.2 Probabilistic metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - BinaryCrossentropy class

    - CategoricalCrossentropy class

    - SparseCategoricalCrossentropy class

    - KLDivergence class

    - Poisson class



1.3 Regression metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - MeanSquaredError class

    - RootMeanSquaredError class

    - MeanAbsoluteError class

    - MeanAbsolutePercentageError class

    - CosineSimilarity class

    - LogCoshError class


1.4 Classification metrics based on True/False positives & negatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - AUC class

    - Precision class

    - Recall class

    - TurePositives class

    - TrueNegatives class

    - FalsePositives class

    - FalseNegatives class

    - PrecisionAtRecall class

    - SensitivityAtSpecificity class

    - SpecificityAtSensitivity class




1.5 image segmentation metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - MeanIoU class


1.6 Hinge metrics for "maximum-margin" Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    - Hinge class

    - SquaredHinge class

    - CategoricalHinge class




2.评价指标的使用——compile() & fit()
------------------------------------




3.评价指标的使用——单独使用
------------------------------------



4.创建自定义评价指标
------------------------------------


5. ``add_metric()`` API
------------------------------------