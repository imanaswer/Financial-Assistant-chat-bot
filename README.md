# ChatBot - Financial  Assistant

This project implements a Python-based chatbot for a financial digital assistant. The chatbot utilizes natural language processing (NLP) techniques and a neural network for intent classification to provide responses related to financial queries and bargaining.

## Features

### Natural Language Processing (NLP)

The chatbot uses NLTK (Natural Language Toolkit) for text preprocessing, which includes:
- **Tokenization**: Breaking down sentences into tokens (words).
- **Stemming**: Reducing words to their root form using Porter Stemmer.

### Neural Network Model

The core of the chatbot is a neural network built using PyTorch:
- **Architecture**: Multi-layer perceptron with fully connected layers.
- **Activation Function**: ReLU (Rectified Linear Unit) for introducing non-linearity.
- **Training**: Trained using a dataset (`data.json`) containing intents and their corresponding patterns.

### Bargaining Functionality

The chatbot includes a unique feature to handle bargaining scenarios:
- **POS Tagging**: Uses NLTK's part-of-speech tagging to identify numeric inputs.
- **Responses**: Provides predefined responses based on the numeric input, simulating negotiation.

## Technologies Used

- **Python**: Programming language used for the entire implementation.
- **PyTorch**: Deep learning framework used to build and train the neural network.
- **NLTK**: Library for natural language processing tasks such as tokenization and stemming.
- **JSON**: Data format used to store intents and responses in `data.json`.

## Requirements

Ensure you have the following installed:
- Python 3.x
- Libraries: numpy, torch, nltk, torchviz



