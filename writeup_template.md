
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
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

I tried the different models but the model adapted from Comma.ai's model worked best. This model consists a Sequential model comprising  cropping, resizing and normalization step with three convolution layers and three fully-connected layers. I trained the model with 20 epoch and the model weights were used for validation purpose.

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
***Specification in Table form***

| Layer  | Detail |
| ------------- |:-------------:| 
|Convolution Layer 1| Filters: 16, Kernel: 8 x 8, Stride: 4 x 4 , Padding: SAME , Activation: ELU |
			

---------

<table>
	<th>Layer</th><th>Details</th>
	<tr>
		<td>Convolution Layer 1</td>
		<td>
			<ul>
				<li>Filters: 16</li>
				<li>Kernel: 8 x 8</li>
				<li>Stride: 4 x 4</li>
				<li>Padding: SAME</li>
				<li>Activation: ELU</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Convolution Layer 2</td>
		<td>
			<ul>
				<li>Filters: 32</li>
				<li>Kernel: 5 x 5</li>
				<li>Stride: 2 x 2</li>
				<li>Padding: SAME</li>
				<li>Activation: ELU</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Convolution Layer 3</td>
		<td>
			<ul>
				<li>Filters: 64</li>
				<li>Kernel: 5 x 5</li>
				<li>Stride: 2 x 2</li>
				<li>Padding: SAME</li>
				<li>Activation: ELU</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Flatten layer</td>
		<td>
			<ul>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Fully Connected Layer 1</td>
		<td>
			<ul>
				<li>Neurons: 512</li>
				<li>Dropout: 0.5</li>
				<li>Activation: ELU</li>
			</ul>
		</td>
	</tr>
   	<tr>
		<td>Fully Connected Layer 2</td>
		<td>
			<ul>
				<li>Neurons: 50</li>
				<li>Activation: ELU</li>
			</ul>
		</td>
	</tr>

	<tr>
		<td>Fully Connected Layer 3</td>
		<td>
			<ul>
				<li>Neurons: 1</li>
				<li>Activation: tanh</li>
			</ul>
		</td>
	</tr>
</table>

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 








-----------------

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I used the data available at Udacity(link). It consist of three types of images: center, right and left as shown below:


| Left          | Center        | Right  |
| ------------- |:-------------:| ------|
|![Left][image2a] | ![Center][image2b] | ![Right][image2c]


Simultanously, I also changed the steering angle for right and left images by 

```
right_info['steering'].apply(lambda x : x-0.2) 
left_info['steering'].apply(lambda x : x+0.2) 
```

Then I increased the data by data augumentation, where I flipped the images using matplotlib's  ```image_f = image.transpose(Image.FLIP_LEFT_RIGHT)``` . A sample of direct and fipped images are as follows:


| Direct         | Flipped       |
| ------------- |:-------------:| 
|![Left][Image2d] | ![Center][Image2e]



While flipping the images, I also changed the steering angles by using :

```
f_steering_angle = -1. * float(item[1][1])
```

After the collection process, I had 43394 number of data points. Initial images are of size ```160,320``` with 3 color chanels. I croped away the top ```64``` rows and bottom ```32``` row of pixels. I then resized this data to ```40x160``` by introducing ```Lambda ``` function and normalized the data by introducing another lambda function.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20. I used an adam optimizer so that manually training the learning rate wasn't necessary.




