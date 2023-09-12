# Audio_classification_CNN

This is a CNN based classification model that classifies audio signals into 10 predefined cllasses.

The dataset used in the model training period is Used From UrbanSoundDataset.
#### https://urbansounddataset.weebly.com/urbansound8k.html

Since i don't have a system with much powerful GPU , I trained my model on a subset of the UrbanSoundDataset.

### Project workflow:
Dataset: The Urbansounddataset has 10 predefined classes , and for each class we have around 900 sample audio.

Loading dataset: Since all audio files are nor=t from same source, we have a data preproxessing step that is done in the load_data.py

Model: A simple VGG net is used for classification application

Result : Results can be seen in the predict_with_CNNNet.py

## How to run:
first load dataset using load_data.py (according to your preference on gpu or cpu)

Create the model class CNNNet (cnn.py)

Train the model on data (trained_cnn.py)

Get prediction results (predict_with_CNNNet.py)
