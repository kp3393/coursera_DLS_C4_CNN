# Coursera Deep Learning Specialization Class 4 (CNN) using TensorFlow 2.x
As access to assignments is lost after finishing the course, this repository will help to go through the assignments on IDE. The aim is to help one brush up on the concepts taught by Professor Andrew Ng.

Summary of each assignment for quick reference:

# C4W1 Assignments

AIM: Functional and sequential implementation of the TENSORFLOW models.

Number of assignments: 2 (Convolution_model_Step_by_Step, Convolution_model_Application). Only the second assignment (Convolution_model_Application) is TensorFlow based and is included in the repository. 

Link to datasets: **[Happy Face](https://www.kaggle.com/datasets/iarunava/happy-house-dataset)**, **[hand signs](https://github.com/kp3393/coursera-deep-learning-specialization/tree/master/C4%20-%20Convolutional%20Neural%20Networks/Week%201/datasets)**

Assignment 1: Binary Classification on **[Happy Face](https://www.kaggle.com/datasets/iarunava/happy-house-dataset)** dataset. 

- Build and train a ConvNet in TF for a binary classification problem using **sequential API.**
- Dataset: The data is stored in **[h5 format](https://docs.h5py.org/en/stable/).** More details can be found [here](https://docs.h5py.org/en/stable/quick.html).
    - Train and Test data is provided.
    - Training dataset has 600 images and images are of shape (64, 64, 3). The batch can be represented as (600, 64, 64, 3).
    - Test dataset has 150 images with same dimensions as train.

Assignment 2: Multiclass Classification on  **[hand signs](https://github.com/kp3393/coursera-deep-learning-specialization/tree/master/C4%20-%20Convolutional%20Neural%20Networks/Week%201/datasets)** dataset.

- Build and train a ConvNet in TF for a multiclass classification using the Functional API.
- Dataset: The data is stored in h5 format like assignment 1.
- Main difference is the use of more flexible functional API and use of one hot encoding.

# C4W2 Assignments

AIM: ResNet model and MobileNet model implementation.

Number of assignments: 2 (Residual Network and MobileNet) 

Link to datasets: **[hand signs](https://github.com/kp3393/coursera-deep-learning-specialization/tree/master/C4%20-%20Convolutional%20Neural%20Networks/Week%201/datasets)**, [Alpaca/Not Alpaca](https://www.kaggle.com/datasets/sid4sal/alpaca-dataset-small)

Assignment 1: ResNet50

- Implement the basic building blocks of ResNets in a deep neural network using Keras.
- Put building blocks (identity block and convolution block) to implement and train a state-of-the-art neural network for image classification.
- Implement a skip connection in your network.
- Dataset: hand signs dataset stored in h5 format. Same as C4W1A1.

Assignment 2: MobileNet

- Create a dataset from a directory.
- Preprocess and augment data using the Sequential API.
- Adapt a pretrained model to new data and train a classifier using the Functional API and MobileNet.
- Fine-tune a classifier's final layers to improve accuracy.
- Fine-tune the final layers of your model to capture high-level details near the end of the network and potentially improve accuracy
- Dataset: The model was trained on [Alpaca/Not Alpaca](https://www.kaggle.com/datasets/sid4sal/alpaca-dataset-small) dataset.
    - The images were stored in two folders names as ‘Alpaca’ and ‘Not Alpaca’. The name of these two filters becomes two classes which are to be classified.

# C4W3 Assignments

AIM: YOLO model and Unet model implementation.

Number of assignments: 2 (YOLO model and Unet model) 

Link to datasets: [drive.ai dataset](https://github.com/kp3393/coursera-deep-learning-specialization/tree/master/C4%20-%20Convolutional%20Neural%20Networks/Week%203/Car%20detection%20for%20Autonomous%20Driving/images), [CARLA dataset](https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip)

Assignment 1: YOLO

- Detect objects in a car detection dataset
- Implement non-max suppression to increase accuracy
- Implement intersection over union
- Handle bounding boxes, a type of image annotation popular in deep learning
- *As the YOLO model is computationally expensive to train, the pre-trained weights are used.*
- Refer this [notebook](https://github.com/kp3393/coursera-deep-learning-specialization/blob/master/C4%20-%20Convolutional%20Neural%20Networks/Week%203/Car%20detection%20for%20Autonomous%20Driving/Autonomous_driving_application_Car_detection.ipynb) for a detailed understanding of the implementation.
- **NOTE:- There was a problem in loading yolo.h5 in the python script. The error was ‘Bad Marshal Data’ error. Refer the following [link](https://stackoverflow.com/questions/63484172/tensorflow-load-data-bad-marshal-data) to see the possible reasons. Project was at TF 2.9.1 and python 3.9.**

Assignment 2: Unet

- Build your own U-Net
- Explain the difference between a regular CNN and a U-net
- Implement semantic image segmentation on the CARLA self-driving car dataset
- Apply sparse categorical crossentropy for pixelwise prediction
- Dataset: [CARLA dataset](https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip)
    1. The [CARLA](https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip) dataset has two subfolders (CameraRGB and CameraSeg). 
    2. The CameraRGB is the actual image dataset. It is of shape (600, 800, 3).
    3. The CameraSeg folder consists of the corresponding masks. It is of the shape (600, 800, 3). However, only the first channel has the data (label) in it. There are 13 different classes in the masked dataset.
- [Here](https://www.kaggle.com/code/oluwatobiojekanmi/carla-image-semantic-segmentation-with-u-net/notebook) is a similar notebook for Image Semantic Segmentation with U-Net on complete CARLA dataset along with splitting of the dataset into training, validation and test dataset.