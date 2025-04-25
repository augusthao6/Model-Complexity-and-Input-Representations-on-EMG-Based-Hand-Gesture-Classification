# Model-Complexity-and-Input-Representations-on-EMG-Based-Hand-Gesture-Classification


We split up the code from our various python notebook (.ipynb) files from Google Colab into python (.py) files for readability


For all tests run:

setup.py - Gathers all imports to set up environment, mounts to grab the necessary Ninapro DB1 files from your Google Drive

functions.py - Makes sure we have all the necessary functions to run classifier model scripts and sort scripts

sort[num].py - Choose to run the file with the # of gestures you want to sort (10, 25, 53), and at very bottom of that file, determine the data you want to extract: time domain (TD), fast fourier transform (FFT), discrete wavelet transform (DWT), or a combination of them.

[model].py - Run this based on which model you want to test: model type (CNN, LSTM, CNN-LSTM), complexity (baseline, simple, or complex cnn), and whether or not it incorporates attention (attention CNN, attention LSTM)

trainingloop.py - This code runs our training loop from our data files gathered from sort[num].py on the model we defined in [model].py. Make sure to change your code and parameters where it defines "model" in line 64 to match. The data is split by the Ninapro DB1 dataset on a 70%-20%-10% split for training, validation, and testing, respectively. We train the model for each of the 27 test subjects for 200 epochs.

Finally, the code saves the average training and validation accuracy results, as well as individual test accuracy results as 2 separate .csv files to your workspace, while additionally uploading them to your specified Google Drive. Be sure to change the name of your files for convenience. You may utilize this data for any visualization or drawing conclusions on your results.
