# Teachable Machine

### Train a convolutional neural network to recognise images, live in the browser.

---

Try it out live in the browser 👉🏻 [(link)](https://francesconatali.com/personalprojects/ML/teachable-machine/)

---

#### How to use it

1. Click on the buttons to gather images while moving an object in front of the webcam. Each class represents a different object. For a better result, try to gather a similar number of images for each class.

2. Click on the "Train & Predict!" button to train the model and start predicting. You can follow the training progress in the console.

3. Click on the "Reset" button to start over, trying different objects and conditions to see how the model performs.

4. Enjoy! 🙂

#### How does it work?

Under the hood, Teachable Machine uses a technique called transfer learning. This means leveraging the power of a pre-trained model to solve a new problem, without needing a lot of data and much quicker.

In this example there are only two classes to be recognised, but the model can be trained for many more if required, and the code in this project can be easily adapted for it.

#### About the model

The model used in this project is MobileNet, a convolutional neural network that is 17 layers deep. The model is trained using Keras, a high-level neural networks API, written in Python and capable of running on top of TensorFlow.

A pre-trained version of the network is loaded directly from TFHub. The pre-trained network is then used to extract high-level features from new images (here taken directly from a webcam) so you can create your very own classifier!

#### Future improvements 🚀

Currently this project only supports images. In the future, I'd like to add support to audio and poses as well. If you'd like to contribute, please feel free to get in touch and/or open a pull request here on GitHub.
