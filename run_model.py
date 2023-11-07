import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb
word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3


def review_encode(s):
    encoded = [1]
    
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(word_index["<UNK>"])
            
    return encoded
            

model = keras.models.load_model("model.keras")

with open("sample_review.txt") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])

# test_review = test_data[0]
# predict = model.predict(np.expand_dims(test_review, axis=0))
# print("Review: ")
# print(decode_review(test_review))
# print("Prediction: " + str(predict[0]))
# print("Actual: " + str(test_labels[0]))
# print(results)