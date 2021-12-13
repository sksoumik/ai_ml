# ai_ml
machine learning, ai, data science notes

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

### Sigmoid Kernel

The function [`sigmoid_kernel`](https://newbedev.com/scikit_learn/modules/generated/sklearn.metrics.pairwise.sigmoid_kernel#sklearn.metrics.pairwise.sigmoid_kernel) computes the sigmoid kernel between two vectors. The sigmoid kernel is also known as hyperbolic tangent, or **Multilayer Perceptron**. 

```python
from sklearn.metrics.pairwise import sigmoid_kernel

# tfv_matrix: vector
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.sigmoid_kernel.html
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
```



