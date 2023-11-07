# Sentiment Analysis with TensorFlow and Keras

This project uses the IMDB dataset to train a sentiment analysis model using TensorFlow and Keras.
Sentiment analysis is the process of determining whether a piece of text is positive or negative. This is a binary classification problem, and the model outputs a probability that the review is positive.

## Files

- `train_model.py`: This script loads the IMDB dataset, preprocesses the data, defines the model, and trains it. The trained model is then saved for later use.

- `run_model.py`: This script loads the trained model, preprocesses new reviews, and uses the model to predict the sentiment of these reviews.

## How to Run

1. Run `train_model.py` to train the model and save it.

```bash
python train_model.py
```

2. Write your reviews in a file named `sample_review.txt`, one review per line.

3. Run `run_model.py` to use the trained model to predict the sentiment of the reviews.

```bash
python run_model.py
```

## Requirements
- Python 3.11.5 or later
- TensorFlow 2.14.0 or later
- Keras 2.14.0 or later
- NumPy 1.26.1 or later

## Model Architecture

The model is a simple neural network with an embedding layer, a global average pooling layer, and two dense layers. The final layer uses a sigmoid activation function to output a probability that the review is positive.

## Data Preprocessing

The IMDB dataset is preprocessed by padding the sequences to a length of 250 words. The words are then mapped to integers, with special tokens for padding, start of sequence, unknown words, and unused words.

