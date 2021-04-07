---
layout: single
title:  "Deep Learning for ship image classification"
date:   2021-03-29
categories: DeepLearning
tags: Python TensorFlow CNN
toc: true
toc_label: "Table of Contents"
toc_icon: "clone"
---

The goal is to create a Deep Learning model using Convolutional Neural Networks that classifies a ship. The Convolutional Neural Network (CNN) is trained with the images of all these types of ships using a 60% of the data and validated simulataneously against 20% of the data and finally test the trained model with the remaining 20% of the data. The model is adjusted with various hyperparameters like using different activation functions, loss functions, changing the epochs, using different neural network initializing methods and finally obtaining the best accurate version of the CNN for the data. The programming is done using TensorFlow API by Google.

#### Data Introduction

The ships dataset from kaggle has around 6252 images which are randomly split into train and test sets for the neural network. The categories of ships and their corresponding codes in the dataset are as follows - {'Cargo': 1, 'Military': 2, 'Carrier': 3, 'Cruise': 4, 'Tankers': 5}. The data can be obtained from this <a href="https://www.kaggle.com/arpitjain007/game-of-deep-learning-ship-datasets">link</a>



#### Solution Approach

1. Firstly, the csv file which has the image name and category of the image is converted to a pandas dataframe and then split into training and testing dataframes.
2. Secondly, the images using the names and labels in the dataframes are converted to an tuple format which specifies images as numpy n-dimensional arrays of shape (32,32) and the labels in the second part of the tuple.
3. Thirdly, the image pixels data that is generated is fed to the neural network made of Conv2D layers, MaxPooling2D, Flatten and Dense layers with appropriate activation functions added to each layer and other methods.
4. Finally the results are evaluated with respect to training, validation losses and accuracies to understand the behavior of the Neural Network, and choose the best conbination of the parameters as the final model.



#### Dependencies

   1. TensorFlow - Refer this <a href="https://www.tensorflow.org/install/pip#system-install">link</a> for installation instructions.
   2. Scikit-learn - Installation instructions <a href="https://scikit-learn.org/stable/install.html">here</a>
   3. NumPy - Installation instructions <a href="https://numpy.org/install/">here</a>
   4. Pandas - Installation instructions <a href="https://pandas.pydata.org/getting_started.html">here</a>
   5. Matplotlib - Installation instructions <a href="https://matplotlib.org/stable/users/installing.html">here</a>

Here I used tensorflow image by Google on Docker, more information about using jupyter notebook with tensorflow dependencies installed can be found <a href = "https://analyticsindiamag.com/docker-solution-for-tensorflow-2-0-how-to-get-started/">here</a>

### Part A - Deep Learning model - Convolutional Neural Network (CNN)

#### Let's import libraries


```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.losses import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display, Image
import scipy
```

#### Loading the Data

It is assumed that data folder is in the same directory as this file. Hence relative path is being in the parameters.


```python
df = pd.read_csv('train.csv', dtype=str)
```


```python
# Viewing the contents of the dataframe

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2823080.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2870024.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2662125.jpg</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2900420.jpg</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2804883.jpg</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



#### Splitting the data into training and testing dataset.


```python
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
```


```python
# Sample images of ships are shown below
for i in range(5):
    display(Image(width=470,filename='./images/'+train_data.image[i]))
```



![jpeg](output_10_0.jpeg)





![jpeg](output_10_1.jpeg)





![jpeg](output_10_2.jpeg)





![jpeg](output_10_3.jpeg)





![jpeg](output_10_4.jpeg)




```python
print("===== Images used in the training data =====", '\n')

display(train_data)

print("===== Images used in the testing data =====", '\n')

display(test_data)
```

    ===== Images used in the training data =====




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>478</th>
      <td>2790315.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5099</th>
      <td>2895143.jpg</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1203</th>
      <td>2677725.jpg</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5674</th>
      <td>2779530.jpg</td>
      <td>2</td>
    </tr>
    <tr>
      <th>142</th>
      <td>2810767.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3772</th>
      <td>2853892.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5191</th>
      <td>2903689.jpg</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5226</th>
      <td>1820874.jpg</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5390</th>
      <td>2884285.jpg</td>
      <td>5</td>
    </tr>
    <tr>
      <th>860</th>
      <td>2903475.jpg</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5001 rows × 2 columns</p>
</div>


    ===== Images used in the testing data =====




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1703</th>
      <td>2525185.jpg</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5448</th>
      <td>2837639.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5058</th>
      <td>2904577.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1149</th>
      <td>2866290.jpg</td>
      <td>3</td>
    </tr>
    <tr>
      <th>432</th>
      <td>2459131.jpg</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>416</th>
      <td>2884436.jpg</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6110</th>
      <td>2782276.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3185</th>
      <td>2843694.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2025</th>
      <td>2792377.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>564</th>
      <td>2838422.jpg</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>1251 rows × 2 columns</p>
</div>


#### Data Preprocessing
Images are to be converted into (32,32) dimensional arrays using ImageDataGenerator() from TensorFlow


```python
data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
```

##### The images data is then split into training, validation and testing sets

What happens here is, the names of the images are in the dataframe and we give the dataframe as the arguemnt to the data_generator.flow_from_dataframe() method that is used for making various adjustments and it is also useful to load the images directly using a single command. So, we convert all the images into 32*32 pixels and define batch size as 64 that is how many images are to be imported and changed simultaneously.


```python
# If this returns an error, it should be either path is wrong or the parameters that are entered are not valid ones.

path_train = './images'

train_df = data_generator.flow_from_dataframe(dataframe=train_data,
                                              directory=path_train,
                                                    x_col='image',
                                                    y_col='category',
                                                    target_size=(32,32),
                                                    class_mode='sparse',
                                                    subset='training',
                                                    batch_size=64,
                                                    seed =1
                                                    )
val_df = data_generator.flow_from_dataframe(dataframe=train_data,
                                            directory=path_train,
                                            x_col='image',
                                                    y_col='category',
                                                    target_size=(32,32),
                                                    class_mode='sparse',
                                                    batch_size=64,
                                                    subset='validation',
                                                    seed =1
                                                    )
test_df = data_generator.flow_from_dataframe(dataframe=test_data,
                                            directory=path_train,
                                            x_col='image',
                                                    y_col='category',
                                                    target_size=(32,32),
                                                    class_mode='sparse',
                                                    batch_size=64,
                                                    seed =1)
```

    Found 4001 validated image filenames belonging to 5 classes.
    Found 1000 validated image filenames belonging to 5 classes.
    Found 1251 validated image filenames belonging to 5 classes.



```python
categories = ['Cargo','Military', 'Carrier', 'Cruise', 'Tankers']
```

#### Formulating our CNN via Object Oriented Programming in Python
1. Model a network
2. Training the model with the given parameters,
3. Plotting the training process and
4. Printing testing results

This part is important since we reuse these functions many times in the project for experimenting with different hyperparameters etc and the functions come in handy as we just need to replace only some parameters


```python
class model_nn:

    def __init__(self, train_df, val_df):
        self.train_df = train_df
        self.val_df = val_df

    def make_cnn_model(self, activation_func='relu', optimizer='adam',
                       loss_func=SparseCategoricalCrossentropy(from_logits=True)):
        self.activation_func = activation_func
        self.optimizer = optimizer
        self.loss_func = loss_func
        """This function takes the activation function,
        optimizer and the loss function for the CNN,
        compiles it and returns the model"""

        model = models.Sequential()
        model.add(layers.Conv2D(
            32, (3, 3), activation=activation_func, input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation=activation_func))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation=activation_func))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation=activation_func))
        model.add(layers.Dense(10))
        model.compile(loss=loss_func, optimizer=optimizer,
                      metrics=['accuracy'])
        self.model = model
        print('Model updated, and the model instance can be retried via object_name.model')
        return model

    @classmethod

    def update_model(self, new_model):
        """This function updates the cnn with any pre-defined model that is provided in the input.
        Use this method only when you have to test a new network architecture like changing layers, initialization etc."""
        self.model = new_model
        return

    def train_model(self, epochs):
        """This function trains the given model with specified number of epochs. It returns the training results"""
        train_steps = self.train_df.n//self.train_df.batch_size
        val_steps = self.val_df.n//self.val_df.batch_size
        self.epochs = epochs
        print(f'Training for {self.epochs} epochs')
        train_results = self.model.fit(self.train_df,
                                  steps_per_epoch=train_steps,
                                  epochs=epochs,
                                  validation_data=self.val_df,
                                  validation_steps=val_steps)
        self.train_results = train_results
        return train_results

    def plot_train_results(self):
        """This function plots and displays the training results (Training Accuracy,
        Validation Accuracy) over the number of epochs"""
        acc = self.train_results.history['accuracy']
        val_acc = self.train_results.history['val_accuracy']

        loss = self.train_results.history['loss']
        val_loss = self.train_results.history['val_loss']
        print('\n')
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='best')
        plt.ylabel('Accuracy', fontsize=15)
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy', fontsize=20)

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='best')
        plt.ylabel('Loss', fontsize=15)
        plt.ylim([0, max(plt.ylim())])
        plt.title('Training and Validation Loss', fontsize=20)
        plt.xlabel('Epoch', fontsize=15)
        plt.show()

    def test_results(self, testing_data):
        """This function return the testing accuracy and loss of the specified model"""
        print('Evaluation of the model on Testing Data')
        test_loss, test_acc = self.model.evaluate(testing_data, verbose=2)
        print('=====================')
        print('Test loss: {:.2f}'.format(test_loss))
        print('Test accuracy: {:.2f}%'.format(test_acc*100))
        print('=====================')
        return
```

### The CNN Model 1:
Parameters:
   * Activation function = ReLU.
   * Loss function = SparseCategoricalCrossentropy()
   * Optimizer = ADAM

More information about ReLU can be found <a href="https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/#:~:text=The%20rectified%20linear%20activation%20function%20or%20ReLU%20for,easier%20to%20train%20and%20often%20achieves%20better%20performance.">here</a>. About Sparse Categorical Cross Entropy, it is <a href="https://leakyrelu.com/2020/01/01/difference-between-categorical-and-sparse-categorical-cross-entropy-loss-function/">here</a> and 'ADAM' optimizer it is <a href="https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/#:~:text=Adam%20is%20an%20optimization%20algorithm%20that%20can%20be,name%20Adam%20is%20derived%20from%20adaptive%20moment%20estimation.">here</a>

The convolution neural network is formed by 3 conv2d layers, 2 MaxPooling and 1 Flatten and 1 Dense layers.


```python
model_obj_1 = model_nn(train_df, val_df)

model_obj_1.make_cnn_model()
print('====== Model Summary =====')
model_obj_1.model.summary()
print('Activation function: ', model_obj_1.activation_func, '\n')
print('Optimizer: ', model_obj_1.optimizer, '\n')
loss_func = 'SparseCategoricalCrossentropy()'
print('Loss funcion: ', loss_func, '\n')
epochs = 10

model_obj_1.train_model(epochs)
model_obj_1.plot_train_results()

model_obj_1.test_results(test_df)
```

    Model updated, and the model instance can be retried via object_name.model
    ====== Model Summary =====
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 30, 30, 32)        896
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 4, 4, 64)          36928
    _________________________________________________________________
    flatten (Flatten)            (None, 1024)              0
    _________________________________________________________________
    dense (Dense)                (None, 64)                65600
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                650
    =================================================================
    Total params: 122,570
    Trainable params: 122,570
    Non-trainable params: 0
    _________________________________________________________________
    Activation function:  relu

    Optimizer:  adam

    Loss funcion:  SparseCategoricalCrossentropy()

    Training for 10 epochs
    Epoch 1/10
    62/62 [==============================] - 9s 141ms/step - loss: 1.7236 - accuracy: 0.3253 - val_loss: 1.4677 - val_accuracy: 0.3573
    Epoch 2/10
    62/62 [==============================] - 8s 129ms/step - loss: 1.4452 - accuracy: 0.3713 - val_loss: 1.3155 - val_accuracy: 0.4677
    Epoch 3/10
    62/62 [==============================] - 8s 126ms/step - loss: 1.2742 - accuracy: 0.4814 - val_loss: 1.1795 - val_accuracy: 0.5417
    Epoch 4/10
    62/62 [==============================] - 9s 138ms/step - loss: 1.1402 - accuracy: 0.5366 - val_loss: 1.0707 - val_accuracy: 0.5771
    Epoch 5/10
    62/62 [==============================] - 8s 126ms/step - loss: 1.0431 - accuracy: 0.5631 - val_loss: 1.0240 - val_accuracy: 0.5854
    Epoch 6/10
    62/62 [==============================] - 8s 128ms/step - loss: 1.0014 - accuracy: 0.6038 - val_loss: 1.1261 - val_accuracy: 0.5323
    Epoch 7/10
    62/62 [==============================] - 8s 129ms/step - loss: 0.9650 - accuracy: 0.6070 - val_loss: 0.9539 - val_accuracy: 0.5958
    Epoch 8/10
    62/62 [==============================] - 8s 126ms/step - loss: 0.9004 - accuracy: 0.6269 - val_loss: 0.9336 - val_accuracy: 0.6208
    Epoch 9/10
    62/62 [==============================] - 8s 133ms/step - loss: 0.8697 - accuracy: 0.6355 - val_loss: 1.0340 - val_accuracy: 0.5917
    Epoch 10/10
    62/62 [==============================] - 9s 142ms/step - loss: 0.8371 - accuracy: 0.6609 - val_loss: 0.8799 - val_accuracy: 0.6385






![png](output_21_1.png)



    Evaluation of the model on Testing Data
    20/20 - 2s - loss: 0.9272 - accuracy: 0.6203
    =====================
    Test loss: 0.93
    Test accuracy: 62.03%
    =====================


#### Changing the activation function

Parameters:

* Activation function = TanH.
* Loss function = SparseCategoricalCrossentropy()
* Optimizer = ADAM


```python
model_obj_2 = model_nn(train_df, val_df)

model_obj_2.make_cnn_model('tanh')
print('====== Model Summary =====')
model_obj_2.model.summary()
print('Activation function: ', model_obj_2.activation_func, '\n')
print('Optimizer: ', model_obj_2.optimizer, '\n')
loss_func = 'SparseCategoricalCrossentropy()'
print('Loss funcion: ', loss_func, '\n')
epochs = 10

model_obj_2.train_model(epochs)
model_obj_2.plot_train_results()

model_obj_2.test_results(test_df)
```

    Model updated, and the model instance can be retried via object_name.model
    ====== Model Summary =====
    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_3 (Conv2D)            (None, 30, 30, 32)        896
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 15, 15, 32)        0
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 13, 13, 64)        18496
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 6, 6, 64)          0
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 4, 4, 64)          36928
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 1024)              0
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                65600
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                650
    =================================================================
    Total params: 122,570
    Trainable params: 122,570
    Non-trainable params: 0
    _________________________________________________________________
    Activation function:  tanh

    Optimizer:  adam

    Loss funcion:  SparseCategoricalCrossentropy()

    Training for 10 epochs
    Epoch 1/10
    62/62 [==============================] - 9s 145ms/step - loss: 1.7068 - accuracy: 0.2922 - val_loss: 1.3597 - val_accuracy: 0.4531
    Epoch 2/10
    62/62 [==============================] - 9s 138ms/step - loss: 1.3153 - accuracy: 0.4704 - val_loss: 1.2086 - val_accuracy: 0.5302
    Epoch 3/10
    62/62 [==============================] - 8s 137ms/step - loss: 1.1669 - accuracy: 0.5349 - val_loss: 1.1140 - val_accuracy: 0.5656
    Epoch 4/10
    62/62 [==============================] - 9s 140ms/step - loss: 1.0695 - accuracy: 0.5641 - val_loss: 1.0476 - val_accuracy: 0.5854
    Epoch 5/10
    62/62 [==============================] - 8s 137ms/step - loss: 0.9668 - accuracy: 0.6118 - val_loss: 0.9565 - val_accuracy: 0.6344
    Epoch 6/10
    62/62 [==============================] - 9s 137ms/step - loss: 0.8873 - accuracy: 0.6519 - val_loss: 0.9289 - val_accuracy: 0.6344
    Epoch 7/10
    62/62 [==============================] - 8s 135ms/step - loss: 0.8387 - accuracy: 0.6748 - val_loss: 0.9166 - val_accuracy: 0.6135
    Epoch 8/10
    62/62 [==============================] - 8s 136ms/step - loss: 0.7714 - accuracy: 0.7045 - val_loss: 0.8978 - val_accuracy: 0.6531
    Epoch 9/10
    62/62 [==============================] - 8s 135ms/step - loss: 0.7016 - accuracy: 0.7367 - val_loss: 0.8675 - val_accuracy: 0.6615
    Epoch 10/10
    62/62 [==============================] - 8s 131ms/step - loss: 0.6400 - accuracy: 0.7652 - val_loss: 0.8333 - val_accuracy: 0.6760






![png](output_23_1.png)



    Evaluation of the model on Testing Data
    20/20 - 2s - loss: 0.9364 - accuracy: 0.6195
    =====================
    Test loss: 0.94
    Test accuracy: 61.95%
    =====================


The tanh activation function gave a similar accuracy than relu function with around 63.14% and the learning speed is similar to relu function.

#### Another Activation function

Parameters:

* Activation function = softsign.
* Loss function = SparseCategoricalCrossentropy()
* Optimizer = ADAM


```python
model_obj_3 = model_nn(train_df, val_df)

model_obj_3.make_cnn_model('softsign')
print('====== Model Summary =====')
model_obj_3.model.summary()
print('Activation function: ', model_obj_3.activation_func, '\n')
print('Optimizer: ', model_obj_3.optimizer, '\n')
loss_func = 'SparseCategoricalCrossentropy()'
print('Loss funcion: ', loss_func, '\n')
epochs = 10

model_obj_3.train_model(epochs)
model_obj_3.plot_train_results()

model_obj_3.test_results(test_df)
```

    Model updated, and the model instance can be retried via object_name.model
    ====== Model Summary =====
    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_6 (Conv2D)            (None, 30, 30, 32)        896
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 15, 15, 32)        0
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 13, 13, 64)        18496
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 6, 6, 64)          0
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 4, 4, 64)          36928
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 1024)              0
    _________________________________________________________________
    dense_4 (Dense)              (None, 64)                65600
    _________________________________________________________________
    dense_5 (Dense)              (None, 10)                650
    =================================================================
    Total params: 122,570
    Trainable params: 122,570
    Non-trainable params: 0
    _________________________________________________________________
    Activation function:  softsign

    Optimizer:  adam

    Loss funcion:  SparseCategoricalCrossentropy()

    Training for 10 epochs
    Epoch 1/10
    62/62 [==============================] - 9s 144ms/step - loss: 1.6830 - accuracy: 0.3131 - val_loss: 1.5312 - val_accuracy: 0.3406
    Epoch 2/10
    62/62 [==============================] - 9s 140ms/step - loss: 1.4845 - accuracy: 0.3627 - val_loss: 1.3359 - val_accuracy: 0.4531
    Epoch 3/10
    62/62 [==============================] - 9s 139ms/step - loss: 1.2618 - accuracy: 0.5003 - val_loss: 1.1868 - val_accuracy: 0.5417
    Epoch 4/10
    62/62 [==============================] - 9s 137ms/step - loss: 1.1252 - accuracy: 0.5492 - val_loss: 1.0784 - val_accuracy: 0.5760
    Epoch 5/10
    62/62 [==============================] - 9s 150ms/step - loss: 1.0368 - accuracy: 0.5672 - val_loss: 1.0547 - val_accuracy: 0.5875
    Epoch 6/10
    62/62 [==============================] - 10s 157ms/step - loss: 0.9902 - accuracy: 0.5966 - val_loss: 0.9759 - val_accuracy: 0.6135
    Epoch 7/10
    62/62 [==============================] - 10s 157ms/step - loss: 0.8887 - accuracy: 0.6386 - val_loss: 0.9597 - val_accuracy: 0.6177
    Epoch 8/10
    62/62 [==============================] - 10s 164ms/step - loss: 0.8758 - accuracy: 0.6435 - val_loss: 0.9101 - val_accuracy: 0.6385
    Epoch 9/10
    62/62 [==============================] - 9s 142ms/step - loss: 0.8170 - accuracy: 0.6744 - val_loss: 0.9059 - val_accuracy: 0.6365
    Epoch 10/10
    62/62 [==============================] - 10s 155ms/step - loss: 0.7471 - accuracy: 0.7048 - val_loss: 0.8867 - val_accuracy: 0.6490






![png](output_26_1.png)



    Evaluation of the model on Testing Data
    20/20 - 2s - loss: 0.9721 - accuracy: 0.6083
    =====================
    Test loss: 0.97
    Test accuracy: 60.83%
    =====================


The softsign activation function gave very less accuracy than tanh function with around 24%  and the learning speed is similar to relu function.

### Changing Loss function
Parameters:
   * Activation function = ReLU.
   * Loss function = categorical_hinge
   * Optimizer = ADAM


```python
# model 4:
model_obj_4 = model_nn(train_df, val_df)

model_obj_4.make_cnn_model(loss_func='categorical_hinge')
print('====== Model Summary =====')
model_obj_4.model.summary()
print('Activation function: ', model_obj_4.activation_func, '\n')
print('Optimizer: ', model_obj_4.optimizer, '\n')
loss_func = 'SparseCategoricalCrossentropy()'
print('Loss funcion: ', loss_func, '\n')
epochs = 10

model_obj_4.train_model(epochs)
model_obj_4.plot_train_results()

model_obj_4.test_results(test_df)
```

    Model updated, and the model instance can be retried via object_name.model
    ====== Model Summary =====
    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_9 (Conv2D)            (None, 30, 30, 32)        896
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 15, 15, 32)        0
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 13, 13, 64)        18496
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 6, 6, 64)          0
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 4, 4, 64)          36928
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 1024)              0
    _________________________________________________________________
    dense_6 (Dense)              (None, 64)                65600
    _________________________________________________________________
    dense_7 (Dense)              (None, 10)                650
    =================================================================
    Total params: 122,570
    Trainable params: 122,570
    Non-trainable params: 0
    _________________________________________________________________
    Activation function:  relu

    Optimizer:  adam

    Loss funcion:  SparseCategoricalCrossentropy()

    Training for 10 epochs
    Epoch 1/10
    62/62 [==============================] - 9s 145ms/step - loss: 0.4408 - accuracy: 0.1101 - val_loss: 0.3820 - val_accuracy: 0.0823
    Epoch 2/10
    62/62 [==============================] - 9s 138ms/step - loss: 0.3803 - accuracy: 0.0877 - val_loss: 0.3858 - val_accuracy: 0.0896
    Epoch 3/10
    62/62 [==============================] - 9s 144ms/step - loss: 0.3829 - accuracy: 0.0847 - val_loss: 0.3813 - val_accuracy: 0.0771
    Epoch 4/10
    62/62 [==============================] - 9s 142ms/step - loss: 0.3830 - accuracy: 0.1060 - val_loss: 0.3788 - val_accuracy: 0.0406
    Epoch 5/10
    62/62 [==============================] - 9s 139ms/step - loss: 0.3671 - accuracy: 0.0958 - val_loss: 0.3749 - val_accuracy: 0.1198
    Epoch 6/10
    62/62 [==============================] - 8s 136ms/step - loss: 0.3803 - accuracy: 0.0862 - val_loss: 0.3796 - val_accuracy: 0.0958
    Epoch 7/10
    62/62 [==============================] - 9s 141ms/step - loss: 0.3781 - accuracy: 0.0834 - val_loss: 0.3786 - val_accuracy: 0.1385
    Epoch 8/10
    62/62 [==============================] - 9s 141ms/step - loss: 0.3665 - accuracy: 0.1065 - val_loss: 0.3679 - val_accuracy: 0.1771
    Epoch 9/10
    62/62 [==============================] - 9s 143ms/step - loss: 0.3714 - accuracy: 0.1154 - val_loss: 0.3701 - val_accuracy: 0.0635
    Epoch 10/10
    62/62 [==============================] - 9s 143ms/step - loss: 0.3728 - accuracy: 0.0944 - val_loss: 0.3666 - val_accuracy: 0.1167






![png](output_29_1.png)



    Evaluation of the model on Testing Data
    20/20 - 2s - loss: 0.3633 - accuracy: 0.0911
    =====================
    Test loss: 0.36
    Test accuracy: 9.11%
    =====================


Parameters:
   * Activation function = ReLU.
   * Loss function = mean_squared_error
   * Optimizer = ADAM


```python
model_obj_5 = model_nn(train_df, val_df)

model_obj_5.make_cnn_model(loss_func='mean_squared_error')
print('====== Model Summary =====')
model_obj_5.model.summary()
print('Activation function: ', model_obj_5.activation_func, '\n')
print('Optimizer: ', model_obj_5.optimizer, '\n')
loss_func = 'SparseCategoricalCrossentropy()'
print('Loss funcion: ', loss_func, '\n')
epochs = 10

model_obj_5.train_model(epochs)
model_obj_5.plot_train_results()
model_obj_5.test_results(test_df)
```

    Model updated, and the model instance can be retried via object_name.model
    ====== Model Summary =====
    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_12 (Conv2D)           (None, 30, 30, 32)        896
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 15, 15, 32)        0
    _________________________________________________________________
    conv2d_13 (Conv2D)           (None, 13, 13, 64)        18496
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 6, 6, 64)          0
    _________________________________________________________________
    conv2d_14 (Conv2D)           (None, 4, 4, 64)          36928
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 1024)              0
    _________________________________________________________________
    dense_8 (Dense)              (None, 64)                65600
    _________________________________________________________________
    dense_9 (Dense)              (None, 10)                650
    =================================================================
    Total params: 122,570
    Trainable params: 122,570
    Non-trainable params: 0
    _________________________________________________________________
    Activation function:  relu

    Optimizer:  adam

    Loss funcion:  SparseCategoricalCrossentropy()

    Training for 10 epochs
    Epoch 1/10
    62/62 [==============================] - 9s 146ms/step - loss: 2.8896 - accuracy: 0.0379 - val_loss: 2.3898 - val_accuracy: 0.0542
    Epoch 2/10
    62/62 [==============================] - 9s 142ms/step - loss: 2.3916 - accuracy: 0.0810 - val_loss: 2.3930 - val_accuracy: 0.1240
    Epoch 3/10
    62/62 [==============================] - 9s 141ms/step - loss: 2.2772 - accuracy: 0.1138 - val_loss: 2.5281 - val_accuracy: 0.1240
    Epoch 4/10
    62/62 [==============================] - 9s 139ms/step - loss: 2.2833 - accuracy: 0.1070 - val_loss: 2.1913 - val_accuracy: 0.1625
    Epoch 5/10
    62/62 [==============================] - 9s 142ms/step - loss: 2.2134 - accuracy: 0.1481 - val_loss: 2.1290 - val_accuracy: 0.0854
    Epoch 6/10
    62/62 [==============================] - 9s 151ms/step - loss: 2.0904 - accuracy: 0.1411 - val_loss: 2.0611 - val_accuracy: 0.1010
    Epoch 7/10
    62/62 [==============================] - 10s 156ms/step - loss: 2.0041 - accuracy: 0.1257 - val_loss: 2.1201 - val_accuracy: 0.2094
    Epoch 8/10
    62/62 [==============================] - 11s 172ms/step - loss: 2.0163 - accuracy: 0.1108 - val_loss: 2.0817 - val_accuracy: 0.1000
    Epoch 9/10
    62/62 [==============================] - 10s 161ms/step - loss: 1.9430 - accuracy: 0.0905 - val_loss: 2.0088 - val_accuracy: 0.0365
    Epoch 10/10
    62/62 [==============================] - 10s 161ms/step - loss: 1.8815 - accuracy: 0.0997 - val_loss: 1.9834 - val_accuracy: 0.1229






![png](output_31_1.png)



    Evaluation of the model on Testing Data
    20/20 - 2s - loss: 2.0929 - accuracy: 0.1423
    =====================
    Test loss: 2.09
    Test accuracy: 14.23%
    =====================


### Changing Epochs


```python
model_obj_6 = model_nn(train_df, val_df)

model_obj_6.make_cnn_model()
print('====== Model Summary =====')
model_obj_6.model.summary()
print('Activation function: ', model_obj_6.activation_func, '\n')
print('Optimizer: ', model_obj_6.optimizer, '\n')
loss_func = 'SparseCategoricalCrossentropy()'
print('Loss funcion: ', loss_func, '\n')
epochs = 15

model_obj_6.train_model(epochs)
model_obj_6.plot_train_results()

model_obj_6.test_results(test_df)
```

    Model updated, and the model instance can be retried via object_name.model
    ====== Model Summary =====
    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_15 (Conv2D)           (None, 30, 30, 32)        896
    _________________________________________________________________
    max_pooling2d_10 (MaxPooling (None, 15, 15, 32)        0
    _________________________________________________________________
    conv2d_16 (Conv2D)           (None, 13, 13, 64)        18496
    _________________________________________________________________
    max_pooling2d_11 (MaxPooling (None, 6, 6, 64)          0
    _________________________________________________________________
    conv2d_17 (Conv2D)           (None, 4, 4, 64)          36928
    _________________________________________________________________
    flatten_5 (Flatten)          (None, 1024)              0
    _________________________________________________________________
    dense_10 (Dense)             (None, 64)                65600
    _________________________________________________________________
    dense_11 (Dense)             (None, 10)                650
    =================================================================
    Total params: 122,570
    Trainable params: 122,570
    Non-trainable params: 0
    _________________________________________________________________
    Activation function:  relu

    Optimizer:  adam

    Loss funcion:  SparseCategoricalCrossentropy()

    Training for 15 epochs
    Epoch 1/15
    62/62 [==============================] - 10s 162ms/step - loss: 1.7071 - accuracy: 0.3243 - val_loss: 1.4468 - val_accuracy: 0.3583
    Epoch 2/15
    62/62 [==============================] - 10s 161ms/step - loss: 1.4036 - accuracy: 0.3998 - val_loss: 1.2628 - val_accuracy: 0.5073
    Epoch 3/15
    62/62 [==============================] - 9s 147ms/step - loss: 1.1909 - accuracy: 0.4987 - val_loss: 1.1057 - val_accuracy: 0.5594
    Epoch 4/15
    62/62 [==============================] - 9s 148ms/step - loss: 1.0696 - accuracy: 0.5755 - val_loss: 1.0743 - val_accuracy: 0.5656
    Epoch 5/15
    62/62 [==============================] - 9s 146ms/step - loss: 0.9954 - accuracy: 0.5835 - val_loss: 1.0165 - val_accuracy: 0.6021
    Epoch 6/15
    62/62 [==============================] - 9s 144ms/step - loss: 0.9636 - accuracy: 0.6049 - val_loss: 0.9606 - val_accuracy: 0.6167
    Epoch 7/15
    62/62 [==============================] - 9s 138ms/step - loss: 0.8744 - accuracy: 0.6518 - val_loss: 0.9085 - val_accuracy: 0.6396
    Epoch 8/15
    62/62 [==============================] - 8s 135ms/step - loss: 0.8371 - accuracy: 0.6674 - val_loss: 0.9223 - val_accuracy: 0.6500
    Epoch 9/15
    62/62 [==============================] - 8s 134ms/step - loss: 0.8014 - accuracy: 0.6812 - val_loss: 0.9146 - val_accuracy: 0.6438
    Epoch 10/15
    62/62 [==============================] - 8s 135ms/step - loss: 0.7534 - accuracy: 0.6939 - val_loss: 0.8646 - val_accuracy: 0.6594
    Epoch 11/15
    62/62 [==============================] - 8s 133ms/step - loss: 0.7094 - accuracy: 0.7199 - val_loss: 0.8520 - val_accuracy: 0.6656
    Epoch 12/15
    62/62 [==============================] - 8s 135ms/step - loss: 0.6727 - accuracy: 0.7381 - val_loss: 0.8355 - val_accuracy: 0.6823
    Epoch 13/15
    62/62 [==============================] - 8s 135ms/step - loss: 0.6699 - accuracy: 0.7372 - val_loss: 0.9001 - val_accuracy: 0.6531
    Epoch 14/15
    62/62 [==============================] - 8s 135ms/step - loss: 0.5967 - accuracy: 0.7722 - val_loss: 0.8752 - val_accuracy: 0.6594
    Epoch 15/15
    62/62 [==============================] - 8s 134ms/step - loss: 0.5870 - accuracy: 0.7718 - val_loss: 0.8709 - val_accuracy: 0.6792






![png](output_33_1.png)



    Evaluation of the model on Testing Data
    20/20 - 2s - loss: 0.9224 - accuracy: 0.6347
    =====================
    Test loss: 0.92
    Test accuracy: 63.47%
    =====================



```python
model_obj_7 = model_nn(train_df, val_df)

model_obj_7.make_cnn_model()
print('====== Model Summary =====')
model_obj_7.model.summary()
print('Activation function: ', model_obj_7.activation_func, '\n')
print('Optimizer: ', model_obj_7.optimizer, '\n')
loss_func = 'SparseCategoricalCrossentropy()'
print('Loss funcion: ', loss_func, '\n')
epochs = 5

model_obj_7.train_model(epochs)
model_obj_7.plot_train_results()

model_obj_7.test_results(test_df)
```

    Model updated, and the model instance can be retried via object_name.model
    ====== Model Summary =====
    Model: "sequential_6"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_18 (Conv2D)           (None, 30, 30, 32)        896
    _________________________________________________________________
    max_pooling2d_12 (MaxPooling (None, 15, 15, 32)        0
    _________________________________________________________________
    conv2d_19 (Conv2D)           (None, 13, 13, 64)        18496
    _________________________________________________________________
    max_pooling2d_13 (MaxPooling (None, 6, 6, 64)          0
    _________________________________________________________________
    conv2d_20 (Conv2D)           (None, 4, 4, 64)          36928
    _________________________________________________________________
    flatten_6 (Flatten)          (None, 1024)              0
    _________________________________________________________________
    dense_12 (Dense)             (None, 64)                65600
    _________________________________________________________________
    dense_13 (Dense)             (None, 10)                650
    =================================================================
    Total params: 122,570
    Trainable params: 122,570
    Non-trainable params: 0
    _________________________________________________________________
    Activation function:  relu

    Optimizer:  adam

    Loss funcion:  SparseCategoricalCrossentropy()

    Training for 5 epochs
    Epoch 1/5
    62/62 [==============================] - 9s 136ms/step - loss: 1.6868 - accuracy: 0.3164 - val_loss: 1.4426 - val_accuracy: 0.4323
    Epoch 2/5
    62/62 [==============================] - 10s 163ms/step - loss: 1.3742 - accuracy: 0.4366 - val_loss: 1.2385 - val_accuracy: 0.5323
    Epoch 3/5
    62/62 [==============================] - 10s 154ms/step - loss: 1.2279 - accuracy: 0.4811 - val_loss: 1.1796 - val_accuracy: 0.5083
    Epoch 4/5
    62/62 [==============================] - 9s 152ms/step - loss: 1.0958 - accuracy: 0.5479 - val_loss: 1.0947 - val_accuracy: 0.5500
    Epoch 5/5
    62/62 [==============================] - 9s 151ms/step - loss: 1.0491 - accuracy: 0.5740 - val_loss: 1.0458 - val_accuracy: 0.5865






![png](output_34_1.png)



    Evaluation of the model on Testing Data
    20/20 - 2s - loss: 1.0851 - accuracy: 0.5548
    =====================
    Test loss: 1.09
    Test accuracy: 55.48%
    =====================


### Changing Gradient Estimation


```python
model_obj_8 = model_nn(train_df, val_df)

model_obj_8.make_cnn_model(optimizer='Adagrad')
print('====== Model Summary =====')
model_obj_8.model.summary()
print('Activation function: ', model_obj_8.activation_func, '\n')
print('Optimizer: ', model_obj_8.optimizer, '\n')
loss_func = 'SparseCategoricalCrossentropy()'
print('Loss funcion: ', loss_func, '\n')
epochs = 10

model_obj_8.train_model(epochs)
model_obj_8.plot_train_results()

model_obj_8.test_results(test_df)
```

    Model updated, and the model instance can be retried via object_name.model
    ====== Model Summary =====
    Model: "sequential_7"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_21 (Conv2D)           (None, 30, 30, 32)        896
    _________________________________________________________________
    max_pooling2d_14 (MaxPooling (None, 15, 15, 32)        0
    _________________________________________________________________
    conv2d_22 (Conv2D)           (None, 13, 13, 64)        18496
    _________________________________________________________________
    max_pooling2d_15 (MaxPooling (None, 6, 6, 64)          0
    _________________________________________________________________
    conv2d_23 (Conv2D)           (None, 4, 4, 64)          36928
    _________________________________________________________________
    flatten_7 (Flatten)          (None, 1024)              0
    _________________________________________________________________
    dense_14 (Dense)             (None, 64)                65600
    _________________________________________________________________
    dense_15 (Dense)             (None, 10)                650
    =================================================================
    Total params: 122,570
    Trainable params: 122,570
    Non-trainable params: 0
    _________________________________________________________________
    Activation function:  relu

    Optimizer:  Adagrad

    Loss funcion:  SparseCategoricalCrossentropy()

    Training for 10 epochs
    Epoch 1/10
    62/62 [==============================] - 10s 151ms/step - loss: 2.2087 - accuracy: 0.3001 - val_loss: 1.9697 - val_accuracy: 0.3438
    Epoch 2/10
    62/62 [==============================] - 9s 148ms/step - loss: 1.8904 - accuracy: 0.3369 - val_loss: 1.7280 - val_accuracy: 0.3448
    Epoch 3/10
    62/62 [==============================] - 9s 140ms/step - loss: 1.6838 - accuracy: 0.3479 - val_loss: 1.6123 - val_accuracy: 0.3458
    Epoch 4/10
    62/62 [==============================] - 9s 145ms/step - loss: 1.6003 - accuracy: 0.3478 - val_loss: 1.5739 - val_accuracy: 0.3365
    Epoch 5/10
    62/62 [==============================] - 9s 145ms/step - loss: 1.5790 - accuracy: 0.3318 - val_loss: 1.5551 - val_accuracy: 0.3427
    Epoch 6/10
    62/62 [==============================] - 9s 144ms/step - loss: 1.5720 - accuracy: 0.3251 - val_loss: 1.5500 - val_accuracy: 0.3417
    Epoch 7/10
    62/62 [==============================] - 9s 139ms/step - loss: 1.5514 - accuracy: 0.3379 - val_loss: 1.5400 - val_accuracy: 0.3458
    Epoch 8/10
    62/62 [==============================] - 9s 139ms/step - loss: 1.5528 - accuracy: 0.3349 - val_loss: 1.5316 - val_accuracy: 0.3479
    Epoch 9/10
    62/62 [==============================] - 9s 141ms/step - loss: 1.5459 - accuracy: 0.3393 - val_loss: 1.5363 - val_accuracy: 0.3396
    Epoch 10/10
    62/62 [==============================] - 9s 139ms/step - loss: 1.5443 - accuracy: 0.3371 - val_loss: 1.5292 - val_accuracy: 0.3469






![png](output_36_1.png)



    Evaluation of the model on Testing Data
    20/20 - 2s - loss: 1.5485 - accuracy: 0.3341
    =====================
    Test loss: 1.55
    Test accuracy: 33.41%
    =====================



```python
model_obj_9 = model_nn(train_df, val_df)

model_obj_9.make_cnn_model(optimizer='Adamax')
print('====== Model Summary =====')
model_obj_9.model.summary()
print('Activation function: ', model_obj_9.activation_func, '\n')
print('Optimizer: ', model_obj_9.optimizer, '\n')
loss_func = 'SparseCategoricalCrossentropy()'
print('Loss funcion: ', loss_func, '\n')
epochs = 10

model_obj_9.train_model(epochs)
model_obj_9.plot_train_results()

model_obj_9.test_results(test_df)
```

    Model updated, and the model instance can be retried via object_name.model
    ====== Model Summary =====
    Model: "sequential_8"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_24 (Conv2D)           (None, 30, 30, 32)        896
    _________________________________________________________________
    max_pooling2d_16 (MaxPooling (None, 15, 15, 32)        0
    _________________________________________________________________
    conv2d_25 (Conv2D)           (None, 13, 13, 64)        18496
    _________________________________________________________________
    max_pooling2d_17 (MaxPooling (None, 6, 6, 64)          0
    _________________________________________________________________
    conv2d_26 (Conv2D)           (None, 4, 4, 64)          36928
    _________________________________________________________________
    flatten_8 (Flatten)          (None, 1024)              0
    _________________________________________________________________
    dense_16 (Dense)             (None, 64)                65600
    _________________________________________________________________
    dense_17 (Dense)             (None, 10)                650
    =================================================================
    Total params: 122,570
    Trainable params: 122,570
    Non-trainable params: 0
    _________________________________________________________________
    Activation function:  relu

    Optimizer:  Adamax

    Loss funcion:  SparseCategoricalCrossentropy()

    Training for 10 epochs
    Epoch 1/10
    62/62 [==============================] - 9s 136ms/step - loss: 1.7897 - accuracy: 0.3472 - val_loss: 1.5309 - val_accuracy: 0.3438
    Epoch 2/10
    62/62 [==============================] - 9s 140ms/step - loss: 1.5167 - accuracy: 0.3302 - val_loss: 1.4746 - val_accuracy: 0.3719
    Epoch 3/10
    62/62 [==============================] - 9s 143ms/step - loss: 1.4522 - accuracy: 0.3624 - val_loss: 1.3823 - val_accuracy: 0.4354
    Epoch 4/10
    62/62 [==============================] - 9s 145ms/step - loss: 1.3629 - accuracy: 0.4291 - val_loss: 1.3459 - val_accuracy: 0.4708
    Epoch 5/10
    62/62 [==============================] - 9s 143ms/step - loss: 1.2975 - accuracy: 0.4665 - val_loss: 1.2556 - val_accuracy: 0.5042
    Epoch 6/10
    62/62 [==============================] - 9s 140ms/step - loss: 1.2466 - accuracy: 0.4999 - val_loss: 1.2141 - val_accuracy: 0.5302
    Epoch 7/10
    62/62 [==============================] - 8s 134ms/step - loss: 1.1891 - accuracy: 0.5084 - val_loss: 1.1632 - val_accuracy: 0.5521
    Epoch 8/10
    62/62 [==============================] - 8s 132ms/step - loss: 1.1511 - accuracy: 0.5264 - val_loss: 1.1205 - val_accuracy: 0.5542
    Epoch 9/10
    62/62 [==============================] - 8s 132ms/step - loss: 1.1008 - accuracy: 0.5618 - val_loss: 1.1289 - val_accuracy: 0.5531
    Epoch 10/10
    62/62 [==============================] - 8s 129ms/step - loss: 1.1052 - accuracy: 0.5379 - val_loss: 1.0846 - val_accuracy: 0.5719






![png](output_37_1.png)



    Evaluation of the model on Testing Data
    20/20 - 2s - loss: 1.1285 - accuracy: 0.5300
    =====================
    Test loss: 1.13
    Test accuracy: 53.00%
    =====================


### Changing Network Initialization - a new model as a whole


```python
initializer = tf.keras.initializers.RandomUniform()

model_9 = models.Sequential()
model_9.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_9.add(layers.Dense(32, kernel_initializer=initializer))
model_9.add(layers.MaxPooling2D((2, 2)))
model_9.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_9.add(layers.MaxPooling2D((2, 2)))
model_9.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_9.add(layers.Flatten())
model_9.add(layers.Dense(64, activation='relu'))
model_9.add(layers.Dense(10, activation='softmax'))
model_9.compile(loss=SparseCategoricalCrossentropy(from_logits=False),
                optimizer='adam', metrics=['accuracy'])

model_obj_10 = model_nn(train_df, val_df)

model_obj_10.update_model(model_9)
print('====== Model Summary =====')
model_obj_10.model.summary()
epochs = 10

model_obj_10.train_model(epochs)
model_obj_10.plot_train_results()

model_obj_10.test_results(test_df)
```

    ====== Model Summary =====
    Model: "sequential_9"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_27 (Conv2D)           (None, 30, 30, 32)        896
    _________________________________________________________________
    dense_18 (Dense)             (None, 30, 30, 32)        1056
    _________________________________________________________________
    max_pooling2d_18 (MaxPooling (None, 15, 15, 32)        0
    _________________________________________________________________
    conv2d_28 (Conv2D)           (None, 13, 13, 64)        18496
    _________________________________________________________________
    max_pooling2d_19 (MaxPooling (None, 6, 6, 64)          0
    _________________________________________________________________
    conv2d_29 (Conv2D)           (None, 4, 4, 64)          36928
    _________________________________________________________________
    flatten_9 (Flatten)          (None, 1024)              0
    _________________________________________________________________
    dense_19 (Dense)             (None, 64)                65600
    _________________________________________________________________
    dense_20 (Dense)             (None, 10)                650
    =================================================================
    Total params: 123,626
    Trainable params: 123,626
    Non-trainable params: 0
    _________________________________________________________________
    Training for 10 epochs
    Epoch 1/10
    62/62 [==============================] - 9s 140ms/step - loss: 1.8687 - accuracy: 0.2508 - val_loss: 1.5404 - val_accuracy: 0.3438
    Epoch 2/10
    62/62 [==============================] - 9s 147ms/step - loss: 1.5618 - accuracy: 0.3111 - val_loss: 1.4401 - val_accuracy: 0.3938
    Epoch 3/10
    62/62 [==============================] - 9s 149ms/step - loss: 1.3943 - accuracy: 0.4158 - val_loss: 1.2782 - val_accuracy: 0.4437
    Epoch 4/10
    62/62 [==============================] - 10s 155ms/step - loss: 1.2199 - accuracy: 0.4957 - val_loss: 1.1668 - val_accuracy: 0.5302
    Epoch 5/10
    62/62 [==============================] - 10s 162ms/step - loss: 1.1145 - accuracy: 0.5446 - val_loss: 1.0582 - val_accuracy: 0.5719
    Epoch 6/10
    62/62 [==============================] - 9s 150ms/step - loss: 1.0428 - accuracy: 0.5579 - val_loss: 1.0396 - val_accuracy: 0.5792
    Epoch 7/10
    62/62 [==============================] - 8s 136ms/step - loss: 0.9843 - accuracy: 0.5983 - val_loss: 0.9813 - val_accuracy: 0.6042
    Epoch 8/10
    62/62 [==============================] - 8s 133ms/step - loss: 0.9229 - accuracy: 0.6096 - val_loss: 0.9309 - val_accuracy: 0.6094
    Epoch 9/10
    62/62 [==============================] - 8s 133ms/step - loss: 0.8583 - accuracy: 0.6322 - val_loss: 0.9028 - val_accuracy: 0.6365
    Epoch 10/10
    62/62 [==============================] - 8s 128ms/step - loss: 0.8475 - accuracy: 0.6497 - val_loss: 0.8999 - val_accuracy: 0.6427






![png](output_39_1.png)



    Evaluation of the model on Testing Data
    20/20 - 2s - loss: 0.9659 - accuracy: 0.5979
    =====================
    Test loss: 0.97
    Test accuracy: 59.79%
    =====================



```python
initializer = tf.keras.initializers.glorot_uniform()

model_10 = models.Sequential()
model_10.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_10.add(layers.Dense(32, kernel_initializer=initializer))
model_10.add(layers.MaxPooling2D((2, 2)))
model_10.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_10.add(layers.MaxPooling2D((2, 2)))
model_10.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_10.add(layers.Flatten())
model_10.add(layers.Dense(64, activation='relu'))
model_10.add(layers.Dense(10, activation='softmax'))
model_10.compile(loss=SparseCategoricalCrossentropy(from_logits=False),
                 optimizer='adam',
                 metrics=['accuracy'])

model_obj_11 = model_nn(train_df, val_df)

model_obj_11.update_model(model_10)
print('====== Model Summary =====')
model_obj_11.model.summary()
epochs = 10

model_obj_11.train_model(epochs)
model_obj_11.plot_train_results()

model_obj_11.test_results(test_df)
```

    ====== Model Summary =====
    Model: "sequential_10"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_30 (Conv2D)           (None, 30, 30, 32)        896
    _________________________________________________________________
    dense_21 (Dense)             (None, 30, 30, 32)        1056
    _________________________________________________________________
    max_pooling2d_20 (MaxPooling (None, 15, 15, 32)        0
    _________________________________________________________________
    conv2d_31 (Conv2D)           (None, 13, 13, 64)        18496
    _________________________________________________________________
    max_pooling2d_21 (MaxPooling (None, 6, 6, 64)          0
    _________________________________________________________________
    conv2d_32 (Conv2D)           (None, 4, 4, 64)          36928
    _________________________________________________________________
    flatten_10 (Flatten)         (None, 1024)              0
    _________________________________________________________________
    dense_22 (Dense)             (None, 64)                65600
    _________________________________________________________________
    dense_23 (Dense)             (None, 10)                650
    =================================================================
    Total params: 123,626
    Trainable params: 123,626
    Non-trainable params: 0
    _________________________________________________________________
    Training for 10 epochs
    Epoch 1/10
    62/62 [==============================] - 10s 152ms/step - loss: 1.7860 - accuracy: 0.2874 - val_loss: 1.4786 - val_accuracy: 0.3521
    Epoch 2/10
    62/62 [==============================] - 9s 143ms/step - loss: 1.4380 - accuracy: 0.4041 - val_loss: 1.2853 - val_accuracy: 0.4760
    Epoch 3/10
    62/62 [==============================] - 9s 145ms/step - loss: 1.2173 - accuracy: 0.5046 - val_loss: 1.1138 - val_accuracy: 0.5906
    Epoch 4/10
    62/62 [==============================] - 9s 146ms/step - loss: 1.0841 - accuracy: 0.5616 - val_loss: 1.0121 - val_accuracy: 0.6146
    Epoch 5/10
    62/62 [==============================] - 9s 149ms/step - loss: 0.9928 - accuracy: 0.5888 - val_loss: 0.9665 - val_accuracy: 0.6240
    Epoch 6/10
    62/62 [==============================] - 10s 155ms/step - loss: 0.9416 - accuracy: 0.6271 - val_loss: 0.9172 - val_accuracy: 0.6302
    Epoch 7/10
    62/62 [==============================] - 12s 188ms/step - loss: 0.8534 - accuracy: 0.6545 - val_loss: 0.9427 - val_accuracy: 0.6115
    Epoch 8/10
    62/62 [==============================] - 11s 174ms/step - loss: 0.8128 - accuracy: 0.6584 - val_loss: 0.8623 - val_accuracy: 0.6500
    Epoch 9/10
    62/62 [==============================] - 9s 152ms/step - loss: 0.7813 - accuracy: 0.6788 - val_loss: 0.8798 - val_accuracy: 0.6604
    Epoch 10/10
    62/62 [==============================] - 9s 151ms/step - loss: 0.7218 - accuracy: 0.7095 - val_loss: 0.9203 - val_accuracy: 0.6302






![png](output_40_1.png)



    Evaluation of the model on Testing Data
    20/20 - 2s - loss: 0.9812 - accuracy: 0.5859
    =====================
    Test loss: 0.98
    Test accuracy: 58.59%
    =====================



```python
initializer = tf.keras.initializers.glorot_normal()

model_11 = models.Sequential()
model_11.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_11.add(layers.Dense(32, kernel_initializer=initializer))
model_11.add(layers.MaxPooling2D((2, 2)))
model_11.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_11.add(layers.MaxPooling2D((2, 2)))
model_11.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_11.add(layers.Flatten())
model_11.add(layers.Dense(64, activation='relu'))
model_11.add(layers.Dense(10, activation='softmax'))
model_11.compile(loss=SparseCategoricalCrossentropy(from_logits=False),
                 optimizer='adam',
                 metrics=['accuracy'])

model_obj_11 = model_nn(train_df, val_df)

model_obj_11.update_model(model_11)
print('====== Model Summary =====')
model_obj_11.model.summary()
epochs = 10

model_obj_11.train_model(epochs)
model_obj_11.plot_train_results()

model_obj_11.test_results(test_df)
```

    ====== Model Summary =====
    Model: "sequential_11"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_33 (Conv2D)           (None, 30, 30, 32)        896
    _________________________________________________________________
    dense_24 (Dense)             (None, 30, 30, 32)        1056
    _________________________________________________________________
    max_pooling2d_22 (MaxPooling (None, 15, 15, 32)        0
    _________________________________________________________________
    conv2d_34 (Conv2D)           (None, 13, 13, 64)        18496
    _________________________________________________________________
    max_pooling2d_23 (MaxPooling (None, 6, 6, 64)          0
    _________________________________________________________________
    conv2d_35 (Conv2D)           (None, 4, 4, 64)          36928
    _________________________________________________________________
    flatten_11 (Flatten)         (None, 1024)              0
    _________________________________________________________________
    dense_25 (Dense)             (None, 64)                65600
    _________________________________________________________________
    dense_26 (Dense)             (None, 10)                650
    =================================================================
    Total params: 123,626
    Trainable params: 123,626
    Non-trainable params: 0
    _________________________________________________________________
    Training for 10 epochs
    Epoch 1/10
    62/62 [==============================] - 10s 152ms/step - loss: 1.7774 - accuracy: 0.2739 - val_loss: 1.5042 - val_accuracy: 0.3479
    Epoch 2/10
    62/62 [==============================] - 10s 163ms/step - loss: 1.4793 - accuracy: 0.3669 - val_loss: 1.3208 - val_accuracy: 0.4458
    Epoch 3/10
    62/62 [==============================] - 10s 155ms/step - loss: 1.2457 - accuracy: 0.4956 - val_loss: 1.1531 - val_accuracy: 0.5417
    Epoch 4/10
    62/62 [==============================] - 9s 149ms/step - loss: 1.1119 - accuracy: 0.5358 - val_loss: 1.0438 - val_accuracy: 0.5969
    Epoch 5/10
    62/62 [==============================] - 9s 149ms/step - loss: 1.0359 - accuracy: 0.5802 - val_loss: 1.0275 - val_accuracy: 0.6021
    Epoch 6/10
    62/62 [==============================] - 9s 143ms/step - loss: 0.9630 - accuracy: 0.6019 - val_loss: 1.0159 - val_accuracy: 0.5844
    Epoch 7/10
    62/62 [==============================] - 9s 146ms/step - loss: 0.9266 - accuracy: 0.6162 - val_loss: 0.9276 - val_accuracy: 0.6240
    Epoch 8/10
    62/62 [==============================] - 10s 155ms/step - loss: 0.8414 - accuracy: 0.6635 - val_loss: 0.9009 - val_accuracy: 0.6333
    Epoch 9/10
    62/62 [==============================] - 9s 151ms/step - loss: 0.8082 - accuracy: 0.6733 - val_loss: 0.8572 - val_accuracy: 0.6615
    Epoch 10/10
    62/62 [==============================] - 9s 144ms/step - loss: 0.7638 - accuracy: 0.7049 - val_loss: 0.8347 - val_accuracy: 0.6792






![png](output_41_1.png)



    Evaluation of the model on Testing Data
    20/20 - 2s - loss: 0.9232 - accuracy: 0.6195
    =====================
    Test loss: 0.92
    Test accuracy: 61.95%
    =====================


### Conclusion

        1. The model is trained with epochs of 15 initially but repeated training of the model in the running kernel makes the model overfit the data and could result in fake accuracies.
        2. With hyperparameter tuning, the model is improved by initialising with Xavier Glorot Initialization namely Xavier Uniform and Xavier Gaussian.
        3. It has been observed that the model performed with other loss functions and optimisers but could yield better results with appropriate batch size
        4. Meanwhile, it is also concluded that using different layers and adding more layers in the model would only increase the comkplexity but not improve the accuray unless an appropriate activation function is given.
        3. Later, the model is tuned with various epochs and batch_size to improve accuracy.
        4. Finally, the CNN gave the maximum accuracy of 68% with initial relu activation function and adam optimiser and no initialization and SparseCategoricalCrossEntropy as the best loss function.

### Author
    Mahesh Kumar Badam Venkata
    Master of Science in Applied Data Science
    Syracuse University, Syracuse, NY


**References:**

1. ADL (24 April 2018), "*An intuitive guide to Convolutional Neural Networks*" retrieved from https://www.freecodecamp.org/news/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050/

2. TensorFlow Tutorials,"*Convolutional Neural Network (CNN)*" retrieved from https://www.tensorflow.org/tutorials/images/cnn

3. Analytics Vidhya Courses, "*Convolutional Neural Networks (CNN) from Scratch*" retrieved from https://courses.analyticsvidhya.com/courses/take/convolutional-neural-networks-cnn-from-scratch/texts/10844923-what-is-a-neural-network

4. TensorFlow Core Documentation, "*Module: tf.keras.initializers*" retrieved from  https://www.tensorflow.org/api_docs/python/tf/keras/initializers?version=nightly



