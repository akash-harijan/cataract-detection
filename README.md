Cataract-Detection
==============================

A simple model of Cataract Detection, using mobilenetV2 as the feature extractor, and fully-connected layers trained on the basis of features.

DataSet
------------

This model was fine-tunned on around 1000 real images, then these images were augmented further to increase data. Augmentation techniques applied can be found at src/data/. Used DataSet will be available online soon.

Installations
-------------

To install the required libraries run this command in main folder

`
pip install -r requirements.txt
`

To test the installed libraries use the testing script at src/models/predict_model.py with the command give below

`
python src/models/predict_model.py
`

It will predict whether given image of eye is a normal eye or it has cataract.



