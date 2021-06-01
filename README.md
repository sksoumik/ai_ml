# ai_ml
machine learning, ai, data science notes

#### Check the tensorflow version

Run

```
python3 -c 'import tensorflow as tf; print(tf.__version__)'
```

#### Tokenization

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

#### Parameter vs Hyperparameter

Parameters are estimated or learned from data. They are not manually set by the practitioners. For example, model **weights** in ANN.  

Hyperparameters are set/specified by the practitioners.  They are often tuned for a given predictive modeling problem. For example, 

- The K in the K-nearest neighbors
- Learning rate
- Batch size
- Number of epochs 





