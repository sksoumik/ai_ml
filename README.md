# ai_ml
machine learning, ai, data science notes

### Interesting Notebooks

1. **[Bringing Old Photos Back to Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)** [Notebook](https://colab.research.google.com/drive/1NEm6AsybIiC5TwTU_4DqDkQO0nFRB-uA?usp=sharing)

2. **[first order model](https://github.com/AliaksandrSiarohin/first-order-model)** Notebook

3. [Get SH\*T Done with PyTorch](https://github.com/curiousily/Getting-Things-Done-with-Pytorch)

4. PyTorch Transformers [Tutorials](https://github.com/abhimishra91/transformers-tutorials)

5. **[huggingtweets](https://github.com/borisdayma/huggingtweets)**

6. Shadow removal from image. [Colab](https://colab.research.google.com/drive/1cJ_dsBUXFaFtjoZB9gDYeahjmysnvnTq)

7. What does a CNN see? [Colab](https://colab.research.google.com/drive/1xM6UZ9OdpGDnHBljZ0RglHV_kBrZ4e-9#scrollTo=ZP9p7mH6RJXp)

### Overfitting vs Underfitting

**Overfitting** happens when your model is too complex. For example, if you are training a deep neural network with a very small dataset like dozens of samples, then there is a high chance that your model is going to overfit. 

**Underfitting** happens when your model is too simple. For example, if your linear regression model trying to learn from a very large data set with hundreds of features. 

### Bias-Variance Trade-off

**Bias** is error due to wrong or overly simplistic assumptions in the learning algorithm you’re using. This can lead to the model underfitting your data, making it hard for it to have high predictive accuracy and for you to generalize your knowledge from the training set to the test set.

**Variance** is error due to too much complexity in the learning algorithm you’re using. This leads to the algorithm being highly sensitive to high degrees of variation in your training data, which can lead your model to overfit the data. You’ll be carrying too much noise from your training data for your model to be very useful for your test data.

The bias-variance decomposition essentially decomposes the learning error from any algorithm by adding the bias, the variance and a bit of irreducible error due to noise in the underlying dataset. Essentially, if you make the model more complex and add more variables, you’ll lose bias but gain some variance — in order to get the optimally reduced amount of error, you’ll have to tradeoff bias and variance. You don’t want either high bias or high variance in your model.

### Check the tensorflow version

Run

```
python3 -c 'import tensorflow as tf; print(tf.__version__)'
```

### Tokenization

Given a character sequence and a defined document unit, tokenization is the task of chopping it up into pieces, called *tokens* , perhaps at the same time throwing away certain characters, such as punctuation. 

Tokens can be either words, characters, or sub-words. Most commonly used tokenization method happens at word level. Pre-trained word embeddings such as word2vec, and GloVe comes under word tokenization. 

Program

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ["I love my dog", "I love my cat"]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
```

Output

```bash
{'i': 1, 'love': 2, 'my': 3, 'dog': 4, 'cat': 5}
```



### Word2Vec



### Parameter vs Hyperparameter

Parameters are estimated or learned from data. They are not manually set by the practitioners. For example, model **weights** in ANN.  



Hyperparameters are set/specified by the practitioners.  They are often tuned for a given predictive modeling problem. For example, 

- The K in the K-nearest neighbors
- Learning rate
- Batch size
- Number of epochs 



### Language Modeling

This is the task of predicting what the next word in a sentence will be based on the history of previous words. The goal of this task is to learn the probability of a
sequence of words appearing in a given language. Language modeling is useful for building solutions for a wide variety of problems, such as speech recognition,
optical character recognition, handwriting recognition, machine translation, and spelling correction.

### Data Lake

A data lake is a centralized repository that allows you to store all your **structured and unstructured data** at any scale.

|    Characteristics    |                        Data Warehouse                        |                          Data Lake                           |
| :-------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|       **Data**        | Relational from transactional systems, operational databases, and line of business applications | Non-relational and relational from IoT devices, web sites, mobile apps, social media, and corporate applications |
|      **Schema**       |  Designed prior to the DW implementation (schema-on-write)   |       Written at the time of analysis (schema-on-read)       |
| **Price/Performance** |       Fastest query results using higher cost storage        |     Query results getting faster using low-cost storage      |
|   **Data Quality **   | Highly curated data that serves as the central version of the truth |    Any data that may or may not be curated (ie. raw data)    |
|       **Users**       |                      Business analysts                       | Data scientists, Data developers, and Business analysts (using curated data) |
|     **Analytics**     |            Batch reporting, BI and visualizations            | Machine Learning, Predictive analytics, data discovery and profiling |



### When to use Precision and Recall as evaluation metric

Precision can be seen as **a measure of quality**, and recall as a measure of **quantity**. 

Higher precision means that an algorithm returns more relevant results than irrelevant ones, and high recall means that an algorithm returns most of the relevant results (whether or not irrelevant ones are also returned).

### When to use F1 as a evaluation metric?

Accuracy is used when the True Positives and True negatives are more important while F1-score is used **when the False Negatives and False Positives** are crucial. Accuracy can be used when the class distribution is similar while F1-score is a better metric when there are **imbalanced classes** .

### Calculate document similarity

Some of the most common and effective ways of calculating similarities are,

***Cosine Distance/Similarity*** - It is the cosine of the angle between two vectors, which gives us the angular distance between the vectors. Formula to calculate cosine similarity between two vectors A and B is,

![img](https://pocket-image-cache.com//filters:format(jpg):extract_focal()/https%3A%2F%2Fmiro.medium.com%2Fmax%2F1144%2F1*YInqm5R0ZgokYXjNjE3MlQ.png)

In a two-dimensional space it will look like this,

![angle between two vectors A and B in 2-dimensional space](https://pocket-image-cache.com//filters:format(jpg):extract_focal()/https%3A%2F%2Fmiro.medium.com%2Fmax%2F1144%2F1*mRjgETrg-mPt8jMBu1VtDg.png)

***Euclidean Distance*** - This is one of the forms of Minkowski distance when p=2. It is defined as follows,

![img](https://pocket-image-cache.com//filters:format(jpg):extract_focal()/https%3A%2F%2Fmiro.medium.com%2Fmax%2F742%2F0*55jbZL3qTdeEI5gL.png)

In two-dimensional space, Euclidean distance will look like this,

![Euclidean distance between two vectors A and B in 2-dimensional space](https://pocket-image-cache.com//filters:format(jpg):extract_focal()/https%3A%2F%2Fmiro.medium.com%2Fmax%2F1144%2F1*aUFcVBD_dBAAayDFfAmo_A.png)

​                                        Fig2:  Euclidean distance between two vectors A and B in 2-dimensional space



### Statistical sampling and Re-sampling

### Compare two images and find the difference between them

The difference between the two images can be measured using Mean Squared Error (MSE) and Structural Similarity Index (SSI).

MSE calculation
```python
def mse(image_A, image_B):
	# NOTE: the two images must have the same dimension
	err = np.sum((image_A.astype("float") - image_B.astype("float")) ** 2)
	err /= float(image_A.shape[0] * image_A.shape[1])
	# return the MSE, the lower the error, the more "similar"
	return err
```

SSI calculation
```python
from skimage.metrics import structural_similarity as ssim
result = ssim(image_A, image_B)
# SSIM value can vary between -1 and 1, where 1 indicates perfect similarity.
```



### Recommender System

Traditionally, recommender systems are based on methods such as clustering, nearest neighbor and matrix factorization. 

**Collaborative filtering:**

Based on past history and what other users with similar profiles preferred in the past.

**Content-based**:

Based on the content similarity. For example, "related articles".  

  

### Sigmoid Kernel

The function [`sigmoid_kernel`](https://newbedev.com/scikit_learn/modules/generated/sklearn.metrics.pairwise.sigmoid_kernel#sklearn.metrics.pairwise.sigmoid_kernel) computes the sigmoid kernel between two vectors. The sigmoid kernel is also known as hyperbolic tangent, or **Multilayer Perceptron**. 

```python
from sklearn.metrics.pairwise import sigmoid_kernel

# tfv_matrix: vector
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.sigmoid_kernel.html
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
```



### Semantic Search

Semantic search is a **data searching technique in a** which a search query aims to not only find keywords, but to determine the **intent and contextual meaning** of the the words a person is using for search. Semantics refer to the philosophical study of meaning.



### What is gradient

A **gradient** is a derivative of a function that has more than one input variable. 

### Data standardization vs Normalization

**Normalization** typically means rescales the values into a **range of [0,1]**. 

**Standardization**: typically means rescales data to have a **mean of 0** and a **standard deviation of 1** (unit variance). 

### Why do we normalize data

For machine learning, every dataset does not require normalization. It is required only when features have different ranges. 

For example, consider a data set containing two features, age(x1), and income(x2). Where age ranges from 0–100, while income ranges from 0–20,000 and higher. Income is about 1,000 times larger than age and ranges from 20,000–500,000. So, these two features are in very different ranges. When we do further analysis, like multivariate linear regression, for example, the attributed income will intrinsically influence the result more due to its larger value. But this doesn’t necessarily mean it is more important as a predictor.

Because different features do not have similar ranges of values and hence **gradients may end up taking a long time** and can oscillate back and forth and take a long time before it can finally **find its way to the global/local minimum**. To overcome the model learning problem, we normalize the data. We make sure that the different features take on similar ranges of values so that **gradient descents can converge more quickly**.

### When Should You Use Normalization And Standardization

**Normalization** is a good technique to use when you do not know the distribution of your data or when you know the distribution is not Gaussian (a bell curve). Normalization is useful when your data has varying scales and the algorithm you are using does not make assumptions about the distribution of your data, such as k-nearest neighbors and artificial neural networks.

**Standardization** assumes that your data has a Gaussian (bell curve) distribution. This does not strictly have to be true, but the technique is more effective if your attribute distribution is Gaussian. Standardization is useful when your data has varying scales and the algorithm you are using does make assumptions about your data having a Gaussian distribution, such as linear regression, logistic regression, and linear discriminant analysis. 

Normalization -> Data distribution is not Gaussian (bell curve).

Standardization -> Data distribution is Gaussian (bell curve). 

### Vanishing Gradient Problem

As the backpropagation algorithm advances downwards(or backward) from the output layer towards the input layer, the gradients often get smaller and smaller and approach zero which eventually leaves the weights of the initial or lower layers nearly unchanged. As a result, the gradient descent never converges to the optimum. This is known as the ***vanishing gradients\*** problem.

###### **Why?**

Certain activation functions, like the sigmoid function, squishes a large input space into a small input space between 0 and 1. Therefore, a large change in the input of the sigmoid function will cause a small change in the output. Hence, the derivative becomes small.

However, when *n* hidden layers use an activation like the sigmoid function, *n* small derivatives are multiplied together. Thus, the gradient decreases exponentially as we propagate down to the initial layers.

###### **Solution**

1. *Use non-saturating activation function:* because of the nature of sigmoid activation function, it starts saturating for larger inputs (negative or positive) came out to be a major reason behind the vanishing of gradients thus making it non-recommendable to use in the **hidden layers** of the network.

   So to tackle the issue regarding the saturation of activation functions like sigmoid and tanh, we must use some other non-saturating functions like ReLu and its alternatives.

2. *Proper weight initialization*: There are different ways to initialize weights, for example, Xavier/Glorot initialization, Kaiming initializer etc. Keras API has default weight initializer for each types of layers. For example, see the available initializers for tf.keras in [keras doc](https://keras.io/api/layers/initializers/#layer-weight-initializers). 

​	You can get the weights of a layer like below:

​      

```python
# tf.keras
model.layers[1].get_weights()
```

  

3. Residual networks are another solution, as they provide residual connections straight to earlier layers. 

4. Batch normalization (BN) layers can also resolve the issue. As stated before, the problem arises when a large input space is mapped to a small one, causing the derivatives to disappear. Batch normalization reduces this problem by simply normalizing the input, so it doesn’t reach the outer edges of the sigmoid function. 

```python
# tf.keras

from keras.layers.normalization import BatchNormalization

# instantiate model
model = Sequential()

# The general use case is to use BN between the linear and non-linear layers in your network, 
# because it normalizes the input to your activation function, 
# though, it has some considerable debate about whether BN should be applied before 
# non-linearity of current layer or works best after the activation function. 

model.add(Dense(64, input_dim=14, init='uniform'))    # linear layer
model.add(BatchNormalization())                       # BN
model.add(Activation('tanh'))                         # non-linear layer
```

Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.

### Why ReLU

 **ReLu is** faster to compute than the **sigmoid** function, and its derivative **is** faster to compute. This makes a significant difference to training and inference time for neural networks. 

Main benefit is that the derivative/gradient of ReLu is either 0 or 1, so multiplying by it won't cause weights that are further away from the end result of the loss function to suffer from the vanishing gradient. 

### What is weight decay

Having fewer parameters is only one way of preventing our model from getting overly complex. But it is actually a very limiting strategy. More parameters mean more interactions between various parts of our neural network. And more interactions mean more non-linearities. These non-linearities help us solve complex problems.

However, we don’t want these interactions to get out of hand. Hence, what if we penalize complexity. We will still use a lot of parameters, but we will prevent our model from getting too complex. This is how the idea of weight decay came up.

One way to penalize complexity, would be to add all our parameters (weights) to our loss function. Well, that won’t quite work because some parameters are positive and some are negative. So what if we add the squares of all the parameters to our loss function. We can do that, however it might result in our loss getting so huge that the best model would be to set all the parameters to 0.

To prevent that from happening, we multiply the sum of squares with another smaller number. This number is called ***weight decay\*** or `wd.`

Our loss function now looks as follows:

```
Loss = MSE(y_hat, y) + wd * sum(w^2)
```

### Mean, Median

[Watch](https://youtu.be/0ifDuw-Qgvo)

### PyTorch Tutorial

See some great resources [here](static/pytorch tutorials.pdf)

### Kaggle

- download kaggle dataset: `kaggle datasets download [username/dataset name (just copy the url after kaggle.com)]`

### NLP Intro

- Concepts of Bag-of-Words (BoW) and TF-IDF come into play. Both BoW and TF-IDF are techniques that help us convert text sentences into **numeric vectors**. [Read](https://www.analyticsvidhya.com/blog/2020/02/quick-introduction-bag-of-words-bow-tf-idf/)
- BERT tokenizer does the preprocessing by itself, so usually you don't benefit from standard preprocessing.
- Transformer models: read [here](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04)
- Common pre-processing techniques:
  1. Removes unicode strings like.
  2. Removes URL strings like.
  3. Removes emoticons from text.
  4. Remove Punctuation (`string.punctuation` or using regular expression).
  5. Convert all words to one case.
  6. Filter out Stop Words (e.g. **Stopwords** are the most common **words** in any natural language. For example, "the”, “is”, “in”, “for”, “where”, “when”, “to”, “at” etc.)
  7. _Stemming:_ A technique that takes the word to its root form. It just removes suffixes from the words. (`nltk.PorterStemmer()` / `nltk.SnowballStemmer()`).

### SVM

Support Vector Machine (SVM) is a supervised machine learning algorithm that can be used for **both classification or regression** challenges. The model extracts the best possible hyper-plane / line that segregates the two classes.

### Random Forest Model

Random Forest models are a type of **ensemble** models, particularly **bagging** models. They are part of the tree-based model family.

### Text classification

###### **Approaches to automatic text classification can be grouped into three categories:**

- Rule-based methods
- Machine learning (data-driven) based methods
- Hybrid methods

###### **neural network architectures, such as models based on**

- recurrent neural networks (RNNs),
- Convolutional neural networks (CNNs),
- Attention,
- Transformers,
- Capsule Nets

### Batch Size

- **Batch Gradient Descent**. Batch size is set to the total number of examples in the training dataset.
- **Stochastic Gradient Descent**. Batch size is set to one.
- **Minibatch Gradient Descent**. Batch size is set to more than one and less than the total number of examples in the training dataset.

### Python’s built-in `sorted()` function

The built-in sorting algorithm of Python uses a special version of merge sort, called Timsort, which runs in O(n log n) on average and worst-case both.

### Multi-class Text Classification

- For multi-class classification: loss-function: categorical cross entropy (For binary classification: binary cross entropy loss).
- BERT: Take a pre-trained BERT model, add an untrained dense layer of neurons, train the layer for any downstream task, …

### Backpropagation

The backward function contains the backpropagation algorithm, where the goal is to essentially minimize the loss with respect to our weights. In other words, the weights need to be updated in such a way that the loss decreases while the neural network is training (well, that is what we hope for). All this magic is possible with the gradient descent algorithm.

### Activation function vs Loss function

An Activation function is a property of the neuron, a function of all the inputs from previous layers and its output, is the input for the next layer.

If we choose it to be linear, we know the entire network would be linear and would be able to distinguish only linear divisions of the space.

Thus we want it to be non-linear, the traditional choice of function (tanh / sigmoid) was rather arbitrary, as a way to introduce non-linearity.

One of the major advancements in deep learning, is using ReLu, that is easier to train and converges faster. but still - from a theoretical perspective, the only point of using it, is to introduce non-linearity. On the other hand, a Loss function, is the goal of your whole network.

it encapsulate what your model is trying to achieve, and this concept is more general than just Neural models.

A Loss function, is what you want to minimize, your error. Say you want to find the best line to fit a bunch of points:

_D_={(*x*1,*y*1),…,(_x\*\*n_,_y\*\*n_)}

$$
D={(x1,y1),…,(x**n,y**n)}
$$

Your model (linear regression) would look like this:

`y=mx+n`

And you can choose several ways to measure your error (loss), for example L1:

or maybe go wild, and optimize for their harmonic loss:

### t-SNE algorithm

(**t**-**SNE**) **t**-Distributed Stochastic Neighbor Embedding is a non-linear dimensionality reduction algorithm **used for** exploring high-dimensional data. It maps multi-dimensional data to two or more dimensions suitable for human observation.

### Cross-Validation data

We should not use augmented data in cross validation dataset.

### Hyper-parameter optimization techniques

- Grid Search
- Bayesian Optimization.
- Random Search

### Normalization in ML

Normalizing helps keep the network weights near zero which in turn makes back-propagation more stable. Without normalization, networks will tend to fail to learn.

### Why do call `scheduler.step()` in pytorch?

If you don’t call it, the learning rate won’t be changed and stays at the initial value.

### Momentum and Learning rate dealing

If the LR is low, then momentum should be high and vice versa. The basic idea of momentum in ML is to **increase the speed of training**.

Momentum helps to know the direction of the next step with the knowledge of the previous steps. It helps to prevent oscillations. A typical choice of momentum is between **0.5 to 0.9**.

### YOLO

You only look once (YOLO) is SOTA real-time object detection system.

### Object Recognition

[Ref link](https://machinelearningmastery.com/object-recognition-with-deep-learning/)

Object classification + Object localization (bbox) = Object detection

Object classification + Object localization + Object detection = Object Recognition (Object detection)

### Target Value Types

Categorical variables can be:

1. Nominal
2. Ordinal
3. Cyclical
4. Binary

Nominal variables are variables that have two or more categories which do not have any kind of order associated with them. For example, if gender is classified into two groups, i.e. male and female, it can be considered as a nominal variable.Ordinal variables, on the other hand, have “levels” or categories with a particular order associated with them. For example, an ordinal categorical variable can be a feature with three different levels: low, medium and high. Order is important.As far as definitions are concerned, we can also categorize categorical variables as binary, i.e., a categorical variable with only two categories. Some even talk about a type called “cyclic” for categorical variables. Cyclic variables are present in “cycles” for example, days in a week: Sunday, Monday, Tuesday, Wednesday, Thursday, Friday and Saturday. After Saturday, we have Sunday again. This is a cycle. Another example would be hours in a day if we consider them to be categories.

### Confusion Matrix

Let's say, we have a dataset which contains cancer patient data (Chest X-ray image), and we have built a machine learning model to predict if a patient has cancer or not.

**True positive (TP):** Given an image, if your model predicts the patient has cancer, and the actual target for that patient has also cancer, it is considered a true positive. Means the prediction is True.

**True negative (TN):** Given an image, if your model predicts that the patient does not have cancer and the actual target also says that patient doesn't have cancer it is considered a true negative. Means the prediction is True.

**False positive (FP):** Given an image, if your model predicts that the patient has cancer but the the actual target for that image says that the patient doesn't have cancer, it a false positive. Means the model prediction is False.

**False negative (FN):** Given an image, if your model predicts that the patient doesn't have cancer but the actual target for that image says that the patient has cancer, it is a false negative. This prediction is also false.

### When not to use accuracy as Metric

If the number of samples in one class outnumber the number of samples in another class by a lot. In these kinds of cases, it is not advisable to use accuracy as an evaluation metric as it is not representative of the data. So, you might get high accuracy, but your model will probably not perform that well when it comes to real-world samples, and you won’t be able to explain to your managers why. In these cases, it’s better to look at other metrics such as precision.

### Common Evaluation Metrics in ML

If we talk about **classification problems**, the most common metrics used are:

- Accuracy

- Precision (P)
- Recall (R)
- F1 score (F1)
- Area under the ROC (Receiver Operating Characteristic) curve or simply AUC (AUC)
- Log loss- Precision at k (P@k)
- Average precision at k (AP@k)
- Mean average precision at k (MAP@k)

When it comes to **regression**, the most commonly used evaluation metrics are:

- Mean absolute error (MAE)
- Mean squared error (MSE)
- Root mean squared error (RMSE)
- Root mean squared logarithmic error (RMSLE)
- Mean percentage error (MPE)
- Mean absolute percentage error (MAPE)- R2

### Autoencoder

An **autoencoder** is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an **autoencoder** is to learn a representation (encoding) for a set of data, typically for dimensionality reduction.

[See](https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder) visual representation and code in pytorch. A great [notebook](https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html) from cstoronto.

##### Difference between AutoEncoder(AE) and Variational AutoEncoder(VAE):

The key difference between and autoencoder and variational autoencoder is autoencoders learn a “compressed representation” of input (could be image,text sequence etc.) automatically by first compressing the input (_encoder_) and decompressing it back (decoder) to match the original input. The learning is aided by using distance function that quantifies the information loss that occurs from the lossy compression. So learning in an autoencoder is a form of unsupervised learning (or self-supervised as some refer to it) - there is no labeled data.

Instead of just learning a function representing the data ( a compressed representation) like autoencoders, variational autoencoders learn the parameters of a probability distribution representing the data. Since it learns to model the data, we can sample from the distribution and generate new input data samples. So it is a generative model like, for instance, GANs.

So, VAE are generative autoencoders, meaning they can generate new instances that look similar to original dataset used for training. VAE learns **probability distribution** of the data whereas autoencoders learns a function to map each input to a number and decoder learns the reverse mapping.

### Why PyTorch?

PyTorch’s clear syntax, streamlined API, and easy debugging make it an excellent choice for introducing deep learning. PyTorch’s dynamic graph structure lets you experiment with _every part of the model_, meaning that the graph and its input can be modified during runtime. This is referred to as **eager execution**. It offers the programmer better access to the inner workings of the network than a static graph (TF) does, which considerably eases the process of debugging the code.

Want to make your own loss function? One that adapts over time or reacts to certain conditions? Maybe your own optimizer? Want to try something really weird like growing extra layers during training? Whatever - PyTorch is just here to crunch the numbers - you drive. [Ref: *Ref: Deep Learning with PyTorch - Eli Stevens*]

### PyTorch vs NumPy

PyTorch is not the only library that deals with multidimensional arrays. NumPy is by far the most popular multidimensional array library, to the point that it has now arguably become the lingua franca of data science. PyTorch features seamless interoperability with NumPy, which brings with it first-class integration with the rest of the scientific
libraries in Python, such as SciPy, Scikit-learn, and Pandas. Compared to NumPy arrays, PyTorch tensors have a few superpowers, such as **the ability to perform very fast operations on graphical processing units (GPUs)**, distribute operations on multiple devices or machines, and keep track of the **graph of computations** that created them.

### Frequently used terms in ML

##### Feature engineering

Features are transformations on input data that facilitate a downstream algorithm, like a classifier, to produce correct outcomes on new data. Feature engineering consists of coming up with the right transformations so that the downstream algorithm can solve a
task. For instance, in order to tell ones from zeros in images of handwritten digits, we would come up with a set of filters to estimate the direction of edges over the image, and then train a classifier to predict the correct digit given a distribution of edge directions. Another useful feature could be the number of enclosed holes, as seen in a zero, an eight, and, particularly, loopy twos. [Read this article](https://medium.com/mindorks/what-is-feature-engineering-for-machine-learning-d8ba3158d97a).

##### Tensor

Tensor is multidimensional arrays similar to NumPy arrays.

##### ImageNet

ImageNet dataset (http://imagenet.stanford.edu). ImageNet is a very large dataset of over 14 million images maintained by Stanford University. All of the images are labeled with a hierarchy of nouns that come from the WordNet dataset (http://wordnet.princeton.edu),
which is in turn a large lexical database of the English language.

##### Embedding

An embedding is a relatively low-dimensional space into which you can translate high-dimensional vectors. The embedding in machine learning or NLP is actually a technique mapping from words to vectors which you can do better analysis or relating, for example, "toyota" or "honda" can be hardly related in words, but in **vector space** it can be set to very close according to some measure, also you can strengthen the relation ship of word by setting: king-man+woman = Queen. So we can set boy to (1,0) and then set girl to (-1,0) to show they are in the same dimension but the meaning is just opposite.

##### Baseline

A baseline is the result of a very basic model/solution. You generally create a baseline and then try to make more complex solutions in order to get a better result. If you achieve a better score than the baseline, it is good.

##### Benchmarking

It a process of measuring the performance of a company's products, services, or processes against those of another business considered to be the best in the industry, aka “best in class.” The point of **benchmarking** is to identify internal opportunities for improvement. The same concept applies for the ML use cases as well. For example, It's a tool, comparing how well one ML method does at performing a specific task compared to another ML method which is already known as the best in that category.

##### Bands and Modes of Image

An image can consist of one or more bands of data. The Python Imaging Library allows you to store several bands in a single image, provided they all have the same dimensions and depth. For example, a PNG image might have ‘R’, ‘G’, ‘B’, and ‘A’ bands for the red, green, blue, and alpha transparency values. Many operations act on each band separately, e.g., histograms. It is often useful to think of each pixel as having one value per band.

The mode of an image defines the **type and depth** of a pixel in the image. The current release supports the following standard modes: [Read](https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html#concept-modes)

##### Mixed-Precision

Mixed precision is the use of both 16-bit and 32-bit floating-point types in a model during training to make it run faster and use less memory. By keeping certain parts of the model in the 32-bit types for numeric stability, the model will have a lower step time and train equally as well in terms of the evaluation metrics such as accuracy.

##### Hyperparameters

With neural networks, you’re usually working with hyperparameters once the data is formatted correctly. A hyperparameter is a parameter whose value is set before the learning process begins. It determines how a network is trained and the structure of the network. Few hyperparameter example:

- Number of hidden layers in the network
- Number of hidden units for each hidden layer
- Learning rate
- Activation function for different layers
- Momentum
- Learning rate decay.
- Mini-batch size.
- Dropout rate (if we use any dropout layer)
- Number of epochs

##### Quantization

Quantization that realizes speeding up and memory saving by replacing the operations of the neural network mainly on floating point operations with integer operations.

This is the most easy way to make inference from trained models which reduce operation costs, reduce calculation loads, and reduce memory consumption.

### Methods for finding out Hyperparameters

1. _Manual Search_

2. _Grid Search_ [(http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/](http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/))

   In scikit-learn there is a `from sklearn.model_selection import GridSearchCV` class to find the best parameters using GridSearch.

   ```python
   from sklearn.model_selection import GridSearchCV
   
   model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
   optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
   param_grid = dict(optimizer=optimizer)
   grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
   ```

   We can't use the gridsearch directly with PyTorch, but there is a library which is called [skorch](https://github.com/skorch-dev/skorch). Using skorch, we can use the sklearns's gridsearch with PyTorch models.

3. _Random Search_

In scikit-learn there is a Class `from sklearn.model_selection import RandomizedSearchCV` which we can use to do random search. We can't use the random search directly with PyTorch, but there is a library which is called [skorch](https://github.com/skorch-dev/skorch). Using skorch, we can use the sklearn's `RandomizedSearchCV` with PyTorch models.

Read [more ...](https://discuss.pytorch.org/t/what-is-the-best-way-to-perform-hyper-parameter-search-in-pytorch/19943)

4. _Bayesian Optimization_

   There are different libraries for searching hyperparameter, for example: optuna, hypersearch. gridsearchCV in sklearn etc.

### If your machine learning model is 99% correct, what are the possible wrong things happened?

1. Overfitting.
2. Wrong evaluation metric
3. Bad validation set
4. Leakage: you're accidentally using 100% of the training set as your test set.
5. Extreme class imbalance (with, say, 98% in one class) combined with the accuracy metric or a feature that leaks the target.

### GAN

GAN, where two networks, one acting as the painter and the other as the art historian, compete to outsmart each other at creating and detecting forgeries. GAN stands for generative adversarial network, where generative means something is being created (in this
case, fake masterpieces), adversarial means the two networks are competing to outsmart the other, and well, network is pretty obvious. These networks are one of the most original outcomes of recent deep learning research. Remember that our overarching goal is to produce synthetic examples of a class of images that cannot be recognized as fake. When mixed in with legitimate examples, a
skilled examiner would have trouble determining which ones are real and which are our forgeries.

The end goal for the generator is to fool the discriminator into mixing up real and fake images. The end goal for the discriminator is to find out when it’s being tricked, but it also helps inform the generator about the identifiable mistakes in the generated images. At the start, the generator produces confused, three-eyed monsters that look nothing like a Rembrandt portrait. The discriminator is easily able to distinguish the muddled messes from the real paintings. As training progresses, information flows back from the discriminator, and the
generator uses it to improve. By the end of training, the generator is able to produce convincing fakes, and the discriminator no longer is able to tell which is which. [ *Ref: Deep Learning with PyTorch - Eli Stevens* ]

##### CycleGAN

An interesting evolution of this concept is the CycleGAN, proposed in 2017. A CycleGAN can turn images of one domain into images of another domain (and back), without the need for us to explicitly provide matching pairs in the training set. It can perform the task of image translation. Once trained you can translate an image from one domain to another domain. For example, when trained on horse and zebra data set, if you give it an image with horses in the ground, the CycleGAN can convert the horses to zebra with the same background. FaceApp is one of the most popular examples of CycleGAN where human faces are transformed into different age groups.

##### StyleGAN

StyleGAN is a GAN formulation which is capable of generating very high-resolution images even of 1024\*1024 resolution. The idea is to build a stack of layers where initial layers are capable of generating low-resolution images (starting from 2\*2) and further layers gradually increase the resolution.

The easiest way for GAN to generate high-resolution images is to remember images from the training dataset and while generating new images it can add random noise to an existing image. In reality, StyleGAN doesn’t do that rather it learn features regarding human face and generates a new image of the human face that doesn’t exist in reality.

##### Text-2-Image

This GAN architecture that made significant progress in generating meaningful images based on an explicit textual description. This GAN formulation takes a textual description as input and generates an RGB image that was described in the textual description.

### Tensors

![](/home/soumik/code/programming_notes/static/tensors.png)

1. https://www.youtube.com/watch?v=otDOHt_Jges&t=617s)

### Reinforcement Learning

An agent interacts with its environment by producing actions and discovers errors or rewards.

### KNN vs K-means clustering

K-Nearest Neighbors is a supervised classification algorithm, while k-means clustering is an unsupervised clustering algorithm. While the mechanisms may seem similar at first, what this really means is that in order for K-Nearest Neighbors to work, you need labeled data you want to classify an unlabeled point into (thus the nearest neighbor part). K-means clustering requires only a set of unlabeled points and a threshold: the algorithm will take unlabeled points and gradually learn how to cluster them into groups by computing the mean of the distance between different points.

The critical difference here is that KNN needs labeled points and is thus supervised learning, while k-means doesn’t—and is thus unsupervised learning.

### ROC curve

The ROC curve is a graphical representation of the contrast between true **positive rates** and the false **positive rate** at various thresholds. It’s often used as a proxy for the trade-off between the **sensitivity** of the model (true positives) vs the fall-out or the probability it will trigger a false alarm (false positives).

### Convolution Operation

There are two inputs to a convolutional operation

i) A 3D volume (input image) of size (nin x nin x channels)

ii) A set of ‘k’ filters (also called as kernels or feature extractors) each one of size (f x f x channels), where f is typically 3 or 5.

An excellent blog post can be found [here](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47).

If you have a stride of 1 and if you set the size of zero padding to

![img](https://adeshpande3.github.io/assets/ZeroPad.png)

where K is the filter size, then the input and output volume **will always have the same spatial dimensions**.

The formula for calculating the output size for any given conv layer is

![img](https://adeshpande3.github.io/assets/Output.png)

where O is the output height/length, W is the input height/length, K is the filter size, P is the padding, and S is the stride.

### Dropout Layers

Dropout layers have a very specific function in neural networks. The problem of overfitting, where after training, the weights of the network are so tuned to the training examples they are given that the network doesn’t perform well when given new examples. The idea of dropout is simplistic in nature. This layer “**drops out” a random set of activations** **in that layer by setting them to zero**. Simple as that. Now, what are the benefits of such a simple and seemingly unnecessary and counterintuitive process? Well, in a way, it forces the network to be redundant. By that I mean the network should be able to provide the right classification or output for a specific example even if some of the activations are dropped out. It makes sure that the network isn’t getting too “fitted” to the training data and thus helps alleviate the overfitting problem. An important note is that this **dropout layer is only used during training, and not during test time.**

### Capsule Networks

[READ MORE...](https://analyticsindiamag.com/why-do-capsule-networks-work-better-than-convolutional-neural-networks/) here.

### L1 and L2 regularization

A regression model that uses L1 regularization technique is called **\*Lasso Regression\*** and model which uses L2 is called **\*Ridge Regression\***.

To implement these two, note that the linear regression model stays the same, but it is the calculation of the **loss function** that includes these regularization terms.

**L1 regularization( Lasso Regression)**- It adds **sum of the** **absolute values** of all weights in the model to cost function. It shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for **feature selection** in case we have a huge number of features.

**L2 regularization(** **Ridge Regression**)- It adds **sum of squares** of all weights in the model to cost function. It is able to learn complex data patterns and gives non-sparse solutions unlike L1 regularization.

In pytorch, we can add these **L2** regularization by adding weight decay parameters.

```python
# adding L2 penalty in the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
```

L1 regularization is not included by default in the PyTorch optimizers, but could be added by including an extra loss `nn.L1Loss` in the weights of the model.

### Semantic segmentation vs Instance Segmentation

Semantic segmentation treats **multiple objects of the same class** as a single entity.

On the other hand, instance segmentation treats **multiple objects of the same class** as distinct individual objects (or **instances**). Typically, **instance segmentation** is harder than **semantic segmentation**.



### Semantic Segmentation

#### **1. Steps to do semantic segmentation**

The goal of semantic image segmentation is to label each **pixel** of an image with a corresponding **class** of what is being represented. Because we’re predicting for every pixel in the image, this task is commonly referred to as **dense prediction**. Thus it is a pixel level image classification.

- The origin could be located at **classification**, which consists of making a prediction for a whole input.
- The next step is **localization / detection**, which provide not only the classes but also additional information regarding the spatial location of those classes.
- Finally, **semantic segmentation** achieves fine-grained inference by making dense predictions inferring labels for every pixel, so that each pixel is labeled with the class of its enclosing object ore region.

#### **2. Generic existing approaches to solve a semantic segmentation problem**

A general semantic segmentation architecture can be broadly thought of as an **encoder** network followed by a **decoder** network:

- The **encoder** is usually is a pre-trained classification network like VGG/ResNet followed by a decoder network.
- The task of the **decoder** is to semantically project the discriminative features (lower resolution) learnt by the encoder onto the pixel space (higher resolution) to get a dense classification.

There are a lot of SOTA architectures to solve this problem. But, U-Net is one of those architectures that stands out, specially for biomedical image segmentation, which use a Fully Convolutional Network Model for the task

Read this blog which explains semantic segmentation and U-Net architecture very well. Link of the [blog](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47)

### Why do we use an activation function?

If we do not have the activation function the weights and bias would simply do a **linear transformation**. A linear equation is simple to solve but is limited in its capacity to solve complex problems and have less power to learn complex functional mappings from data. A neural network without an activation function is just a linear regression model.

Activation function is nothing but a mathematical function that takes in an input and produces an output. The function is activated when the computed result reaches the specified threshold.

Activation functions can add non-linearity to the output. Subsequently, this very feature of activation function makes neural network solve non-linear problems. Non-linear problems are those where there is no direct linear relationship between the input and output.

Some of the non-linear activation functions are: Sigmoid, ReLU, TenH, Softmax etc.

**Sigmoid**

The output of the sigmoid function always ranges between 0 and 1. Sigmoid is very popular in classification problems.

**RELU**

ReLU is one of the most used activation functions. It is preferred to use RELU in the hidden layer. The concept is very straight forward. It also adds non-linearity to the output. However the result can range from 0 to infinity. If you are unsure of which activation function you want to use then use RELU. The main reason why ReLu is used is because it is simple, fast, and empirically it seems to work well.

ReLU (or Rectified Linear Unit) is the most widely used activation function. It gives an **output of X if X is positive and zeros otherwise**. ReLU is often used for hidden layers.

**Softmax Activation Function**

Softmax is an extension of the Sigmoid activation function. Softmax function adds non-linearity to the output, however it is mainly used for classification examples where multiple classes of results can be computed. Softmax is an activation function that generates the output between **zero and one**. It divides each output, such that the total sum of the outputs is equal to one. Softmax is often used for output layers.

### Sigmoid activation function

Sigmoid activation function makes sure that mask pixels are in [0, 1] range.

- **ReLU** is used in the hidden layers.
- **Sigmoid** is used in the output layer while making **binary predictions.**
- **Softmax** is used in the output layer while making **multi-class predictions.**

### Evaluation metrics

##### Semantic segmentation:

IoU, Jaccard Index (Intersection-Over-Union) are mostly used.

![image](/home/soumik/code/programming_notes/static/iou.png)

IoU is the **area of overlap** between the **predicted segmentation** and the **ground truth** divided by the **area of union** between the **predicted segmentation** and the **ground truth**, as shown on the image to the left. This metric ranges from 0–1 (0–100%) with 0 signifying no overlap and 1 signifying perfectly overlapping segmentation.

In scikit-learn there is a built-in function to calculate Jaccard index (IoU): Say, we have

```
 predictions   |   true_label

 0|0|0|1|2         0|0|0|1|2
 0|2|1|0|0         0|2|1|0|0
 0|0|1|1|1         0|0|1|1|1
 0|0|0|0|1         0|0|0|0|1
```

Then, we can do the following:

```python
from sklearn.metrics import jaccard_similarity_score
jac = jaccard_similarity_score(predictions, label, Normalize = True/False)
```

##### Instance Segmentation

TODO: 

### What does the decoder do?

Autoencoders are widly used with the image data and some of their use cases are:

- Dimentionality Reduction
- Image Compression
- Image Denoising
- Image Generation
- Feature Extraction

Encoder-decoder (ED) architecture works well for short sentences, but if the text is too long (maybe higher than 40 words), then the ED performance comes down.

### Why CNN works?

Cause they try to find patterns in input data. Convolutional neural networks work because it's a good extension from the standard deep-learning algorithm.

Given unlimited resources and money, there is no need for convolutional because the standard algorithm will also work. However, convolutional is more efficient because it **reduces the number of parameters**. The reduction is possible because it takes advantage of **feature locality**.

### Transfer Learning

##### One shot vs few shot:

It's about few/one/zero examples in _transfer learning_ to new data after being trained on a dataset that's generally much larger.

For an example, if you train on a dataset that has a million cat pictures, a million dog pictures, and a million horse pictures, and ask it to identify cats/dogs/horses, that's normal supervised learning.

Then you give one example of a crocodile picture (in addition to the above mentioned millions of cats/dogs/horses) and ask the system to identify crocodiles, that's one-shot learning.

##### Zero-Shot Learning

To me, this is the most interesting sub-field. With zero-shot learning, the target is to classify unseen classes without a single training example.

How does a machine “learn” without having any data to utilize?

Think about it this way. Can you classify an object without ever seeing it?

Yes, you can if you have adequate information about its appearance, properties, and functionality. Think back to how you came to understand the world as a kid. You could spot Mars in the night sky after reading about its color and where it would be that night, or identify the constellation Cassiopeia from only being told “it’s basically a malformed ‘W’”.

According to this year trend in NLP, [Zero shot learning will become more effective](https://blog.floydhub.com/ten-trends-in-deep-learning-nlp/#9-zero-shot-learning-will-become-more-effective).

A machine utilizes the **metadata** of the images to perform the same task. **The metadata is nothing but the features associated with the image**.

##### Knowledge transfer in Transfer Learning (TL)

In TL (Transfer Learning): "Knowledge" in is trained **model weights**.

In NLU (Natural Language Understanding): Knowledge refers to structured data.

### Sequence modeling (NLP)

First step is always generating a vector from text.

Classic way of doing this thing is using **Bag-of-Words**.

- One dimension per word in vocabulary.

But can't work well for ordering. This can be also solved using N-grams. But the dimensionality becomes very high.

**RNN**

### Statistics Resources

1. Gaussian/Normal distribution - [[YouTube - Krish Naik](https://youtu.be/UQOTNkq0X48)]

### How to compute the similarity between two text documents?

The common way of doing this is to transform the documents into TF-IDF vectors and then compute the cosine similarity between them.

TF-IDF: Convert a collection of raw documents to a matrix of TF-IDF features. **tf–idf**, **TF\*IDF**, or **TFIDF**, short for **term frequency–inverse document frequency**, is a numerical statistic that is intended to reflect how important a word is to a [document](https://en.wikipedia.org/wiki/Document) in a collection or [corpus](https://en.wikipedia.org/wiki/Text_corpus). This is a technique to quantify a word in documents, we generally compute a weight to each word which signifies the importance of the word in the document and corpus. This method is a widely used technique in Information Retrieval and Text Mining.

CountVectorizer is another technique that can do word count of the words in each document.

Cosine Similarity: Cosine similarity is a metric used to determine how similar the documents are irrespective of their size. Mathematically, it measures the **cosine of the angle** between two vectors projected in a multi-dimensional space. In this context, the two vectors I am talking about are arrays containing the word counts of two documents.

### Common pre-processing steps for NLP

1. Removing punctuations.
2. Normalizing case (lower/upper)
3. Filter out stop words (i', 'me', 'my', 'myself', 'we', 'our' etc)
4. Stemming
5. Lemmitization.
   6.

### Model Re-Training | Continuous model deployment

1. **Model drift**: model deployment should be treated as a continuous process. Rather than deploying a model once and moving on to another project, machine learning practitioners need to retrain their models if they find that the data distributions have deviated significantly from those of the original training set. This concept, known as **model drift**.
2. **Monitoring of continual learning pipelines**: There are great tools in Kubernetes, or Prometheus alongside AlertManager that you can use to monitor all the input data. And you should utilize cloud services and Kubernetes to automate your machine learning infrastructure and experimentation.
3. If you decide to retrain your model periodically, then batch retraining is perfectly sufficient. This approach involves scheduling model training processes on a recurring basis using a **job scheduler** such as Jenkins or [Kubernetes CronJobs](https://mlinproduction.com/k8s-cronjobs/). If you’ve automated model drift detection, then it makes sense to trigger model retraining when drift is identified.
4. We can also use amazon SageMaker for managing the ML infrastructure.

### Image denoising

Commonly used in image denoising:

- convolutional neural network
- pulse coupled neural network
- wavelet neural network

### What is the difference between Image Processing and Computer Vision

In image processing, an image is "processed", that is, transformations are applied to an input image and an output image is returned. The transformations can e.g. be "smoothing", "sharpening", "contrasting" and "stretching". The transformation used depends on the context and issue to be solved.

In computer vision, an image or a video is taken as input, and the goal is to understand (including being able to infer something about it) the image and its contents. Computer vision uses image processing algorithms to solve some of its tasks.

The main difference between these two approaches are the **goals** (not the methods used). For example, if the goal is to enhance an image for later use, then this may be called image processing. If the goal is to emulate human vision, like object recognition, defect detection or automatic driving, then it may be called computer vision.

So basically, Image processing is related to enhancing the image and play with the features like colors. While computer vision is related to "Image Understanding".

### ARIMA vs LSTM for time-series data

Read [here](https://www.datasciencecentral.com/profiles/blogs/arima-sarima-vs-lstm-with-ensemble-learning-insights-for-time-ser)

### What is difference between Random Forest and Decision Trees?

Two concepts are similar. As is implied by the names "Tree" and "Forest," a Random Forest is essentially a collection of Decision Trees. A decision tree is built on an entire dataset, using all the features/variables of interest, whereas a random forest randomly selects observations/rows and specific features/variables to build multiple decision trees from and then averages the results. After a large number of trees are built using this method, each tree "votes" or chooses the class, and the class receiving the most votes by a simple majority is the "winner" or predicted class. There are of course some more detailed differences, but this is the main conceptual difference.

When using a decision tree model on a given training dataset the accuracy keeps improving with more and more splits. You can easily overfit the data and doesn't know when you have crossed the line unless you are using cross validation (on training data set). The advantage of a simple decision tree is model is easy to interpret, you know what variable and what value of that variable is used to split the data and predict outcome.

A random forest is like a black box and works as mentioned in above answer. It's a forest you can build and control. You can specify the number of trees you want in your forest(n_estimators) and also you can specify max num of features to be used in each tree. But you cannot control the randomness, you cannot control which feature is part of which tree in the forest, you cannot control which data point is part of which tree. Accuracy keeps increasing as you increase the number of trees, but becomes constant at certain point. Unlike decision tree, it won't create highly biased model and reduces the variance.

When to use to decision tree:

1. When you want your model to be simple and explainable
2. When you want non parametric model
3. When you don't want to worry about feature selection or regularization or worry about multi-collinearity.
4. You can overfit the tree and build a model if you are sure of validation or test data set is going to be subset of training data set or almost overlapping instead of unexpected.

When to use random forest :

1. When you don't bother much about interpreting the model but want better accuracy.
2. Random forest will reduce variance part of error rather than bias part, so on a given training data set decision tree may be more accurate than a random forest. But on an unexpected validation data set, Random forest always wins in terms of accuracy.

### What's the best way to initialize the weights of a neural network?

No one really knows. Thought experiment: an optimal initialization would in theory perform best at the task in question for a given architecture. But that would be task-specific, so it would depend on the dataset and the desired output. So not a general solution.

### Does the image format (png, jpg, gif) affect how an image recognition neural net is trained?

Short answer is **NO**.

The format in which the image is encoded has to do with its quality. Neural networks are essentially mathematical models that perform lots and lots of operations (matrix multiplications, element-wise additions and mapping functions). A neural network sees a [Tensor](https://en.wikipedia.org/wiki/Tensor) as its input (i.e. a multi-dimensional array). It's shape usually is 4-D (number of images per batch, image height, image width, number of channels).

Different image formats (especially lossy ones) may produce different input arrays but strictly speaking neural nets see arrays in their input, and _NOT_ images.

### What to do when there is no data/little data for a ML product

Consider the task of building a chatbot or text classification system at your organization. In the beginning there may be little or no data to work with. At this point, a basic solution using rule-based systems or traditional machine learning will be apt. As you accumulate more data, more sophisticated NLP techniques (which are often data intensive) can be used, including deep learning. At each step of this journey there are dozens of alternative approaches one can take. 

### Deal with imbalance data

See [this](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) tutorial. 



