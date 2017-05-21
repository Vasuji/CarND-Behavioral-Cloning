
# Behavioral Cloning

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2a]: ./examples/left.jpg "left"
[image2b]: ./examples/center.jpg "center"
[image2c]: ./examples/right.jpg "right"
[image2d]: ./examples/left_d.jpg "direct"
[image2e]: ./examples/left_f.jpg "flipped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

-----

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.









---------

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I tried the different models but the model adapted from [Comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py)'s model worked best. This model consists a Sequential model comprising of cropping, resizing and normalization step with three convolution layers and three fully-connected layers. I trained the model with 20 epoch and the model weights were used for test.

The model code are shown below :

```
def get_model(verbose):

    # Model adapted from Comma.ai model

    model = Sequential()

    # Crop 64 pixels from the top of the image and 32 from the bottom
    model.add(Cropping2D(input_shape=(160, 320, 3),
                         cropping=((64, 32), (0, 0)),
                         data_format="channels_last"))
    
    # resize the images to 40x160
    model.add(Lambda(resize))
    
    # Normalise the data
    model.add(Lambda(lambda x: (x/255.0) - 0.5))

    # Conv layer 1
    model.add(Convolution2D(16, (8, 8), padding="same", strides=(4, 4)))
    model.add(ELU())

    
    # Conv layer 2
    model.add(Convolution2D(32, (5, 5), padding="same", strides=(2, 2)))
    model.add(ELU())

    # Conv layer 3
    model.add(Convolution2D(64, (5, 5), padding="same", strides=(2, 2)))
              

    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())

    # Fully connected layer 1
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())

    # Fully connected layer 2
    model.add(Dense(50))
    model.add(ELU())

    model.add(Dense(1))

    adam = Adam(lr=0.0001)

    model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
    if verbose:
        print("Model summary:\n", model.summary())
    return model

```

---------


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 91,96). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. [test video](https://github.com/Vasuji/CarND-Behavioral-Cloning-P3/blob/master/run1.mp4) is available in the repository.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 105).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and flipped images.

For details about how I created the training data, see the next section. 

-----------------







### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to develop a model from data so that we could drive a car in autonomous mode. Following are the steps in solution design aproach:

***Convolution***

   My first step was to use a convolution neural network model similar to the [Comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py). I thought this model might be appropriate because model building needs different property of image from different deepth of neural network. More strikingly it has to be precise on decision making for example: If the car is to the side of the lane, it should steer differently than if it is in the center of a lane. Central idea is where is car from middle of two lanes.

   
***Fully connected Layesr***

  I added a fully connected layer after the convolutions to allow the model to perform high-level reasoning on the features taken from the convolutions.
  
***Final Layer***

  This is a regression and not a classification problem since the output (steering angle) is continuous, ranging from -1.0 to 1.0. So instead of ending with a softmax layer, I used a 1-neuron fully connected layer as my final layer. 

***Battle with overfitting***

   In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

  To combat the overfitting, I modified the model  and added more dropout layers. The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track when I tried with centre images only. To improve the driving behavior in these cases, I used all center,right,left and their fliped version of images.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 77-88) consisted of a convolution neural network with the following layers and layer sizes.

***Specification in Table form***

| Layer  | Detail |
| ------------- |:-------------:| 
|Convolution Layer 1| Filters: 16, Kernel: 8 x 8, Stride: 4 x 4 , Padding: SAME , Activation: ELU |
|Convolution Layer 2| Filters: 32, Kernel: 5 x 5, Stride: 2 x 2 , Padding: SAME , Activation: ELU |
|Convolution Layer 3| Filters: 64, Kernel: 5 x 5, Stride: 2 x 2 , Padding: SAME , Activation: ELU |
|Flatten Layer|   |
|Fully connected layer 1| Neurons: 512, Dropout: 0.5, Activation: ELU |
|Fully connected layer 2| Neurons: 50, Activation: ELU |

Here is a visualization of the architecture with its parameters. 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I used the data made available by Udacity. It consist of three types of images: center, right and left as shown below:


| Left          | Center        | Right  |
| ------------- |:-------------:| ------|
|![Left][image2a] | ![Center][image2b] | ![Right][image2c]


Simultanously, I also changed the steering angle for right and left images by adding and substracting correction (0.15)

```
right_info['steering'].apply(lambda x : x-0.15) 
left_info['steering'].apply(lambda x : x+0.15) 
```

Then I increased the data by data augumentation, where I flipped the images using matplotlib's  ```image_f = image.transpose(Image.FLIP_LEFT_RIGHT)``` . A sample of direct and fipped images are as follows:


| Direct         | Flipped       |
| ------------- |:-------------:| 
|![Left][Image2d] | ![Center][Image2e]



While flipping the images, I also changed the steering angles by multiplying old one by -1.0, which means there is rotation by 180 degree or mirror reflection.

```
f_steering_angle = -1. * float(item[1][1])
```

After the collection process, I had 43394 number of data samples. Initial images are of size ```160,320``` with 3 color chanels. I croped away the top ```64``` rows and bottom ```32``` row of pixels. I then resized this data to ```40x160``` by introducing ```Lambda ``` function and normalized the data by introducing another lambda function.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20. I used an adam optimizer so that manually training the learning rate wasn't necessary.




