# Image Classification using Convolutional Neural Network
The aim in this project is to classify the images of people wearing masks and not wearing masks using Convolutional Neural Network(CNN).
**The workflow in this project is as below:**
### Understanding and Pre-processing Data
Before we can start to predict, we first need to understand the dataset we are going to use. The next step is to process the data so that it can be used to train the model, some of the data processing performed is as follows:
- Read images from dataset.
- Resize images to the same size to make the model predictions are more accurate.
- Separate the data into train_data and validation_data.
- Data augmentation to expand the training data by creating variations in the image.

### Create the Model
The model consists of four convolution blocks with max pool layer in each of them, activated by 'ReLU' activation function and 'Sigmoid' activation function in the output layer.
And then in the process of compiling the model we use 'Adam' optimizer and 'Binary Crossentropy' for the loss function.

### Train the Model

### Test the Model

### Evaluate the Model

### Try to Predict the Image
