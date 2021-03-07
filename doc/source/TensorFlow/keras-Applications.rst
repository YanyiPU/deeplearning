
Keras Applications
========================

1.Keras 介绍
----------------------------

Keras Applications are deep learning models that are made available alongside pre-trained 
weights. These models can be used for prediction, feature extraction, and fine-tuning.

Weights are downloaded automatically when instantiating a model. They are stored at 
``~/.keras/models/``.

Upon instantiation, the models will be built according to the image data format set in 
your Keras configuration file at ~/.keras/keras.json. For instance, if you have set 
``image_data_format=channels_last``, then any model loaded from this repository will 
get built according to the TensorFlow data format convention, "Height-Width-Depth".

    - ~/.keras/models/

    - ~/.keras/keras.json


2.目前可用的 App
----------------------------

    - Xception
    - EfficientNet B0 to B7
    - VGG16 and VGG19
    - ResNet and ResNetV2
    - MobileNet and MobileNetV2
    - DenseNet
    - NasNetLarge and NasNetMobile
    - InceptionV3
    - InceptionResNetV2




