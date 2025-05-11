# Install TensorFlow (only needed if not already installed)
#!pip install tensorflow
#pip install wrapt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.model_selection import train_test_split
import numpy as np
# Sample classification data
texts = [
    "The mitochondria is the powerhouse of the cell",  # Science
    "Newton's laws explain motion",                    # Science
    "The French Revolution began in 1789",             # History
    "Shakespeare wrote many plays",                    # English
    "The area of a triangle is half base times height",# Math
    "E=mc^2 is a famous equation"                      # Science
]
labels = ["Science", "Science", "History", "English", "Math", "Science"]

# Encode labels to integers
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
y = np.array(label_tokenizer.texts_to_sequences(labels)) - 1  # to start from 0
y = to_categorical(y)

# Tokenize texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vocab_size = len(tokenizer.word_index) + 1
num_classes = y.shape[1]

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=16, input_length=X.shape[1]),
    SimpleRNN(16),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
# Sample text corpus (History topic)
text_corpus = """
The Renaissance was a period of great cultural change and achievement in Europe.
It spanned from the 14th to the 17th century and marked the transition from the Middle Ages to modernity.
"""

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_corpus])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
words = tokenizer.texts_to_sequences([text_corpus])[0]

for i in range(1, len(words)):
    n_gram_sequence = words[:i+1]
    input_sequences.append(n_gram_sequence)

# Pad sequences
max_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)
model_gen = Sequential([
    Embedding(input_dim=total_words, output_dim=16, input_length=max_len - 1),
    SimpleRNN(32),
    Dense(total_words, activation='softmax')
])

model_gen.compile(loss='categorical_crossentropy', optimizer='adam')
model_gen.fit(X, y, epochs=100, verbose=1)

def generate_text(seed_text, next_words=5):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')
        predicted_probs = model_gen.predict(token_list, verbose=0)
        predicted = np.argmax(predicted_probs)
        
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += ' ' + output_word
    return seed_text

print(generate_text("The Renaissance ", next_words=20))
