import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import load_data, TextProcessor, convert_text_to_tensors
from torch.utils.data import DataLoader

#########################################################
# COMP331 Fall 2025 PA2
# This file contains the model class, training loop and evaluation function 
# for you to implement
#########################################################


class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, max_length=20):
        super(NeuralNetwork, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        #######################
        # TODO: Define your model class 
        # You must include an embedding layer, 
        # at least one linear layer, and an activation function
        # you may change inputs to the init method as you want
        #######################

        #create the embedding layer, our word embeddings for our input
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        #create the first hidden layer, takes the embeddings in and outputs the hidden layer sized output
        self.linear1 = nn.Linear(self.embedding_dim, self.hidden_size)
        #relu layer takes any input and returns a = 0 or >0 value
        self.relu = nn.ReLU()
        #second hidden layer, takes hidden in outputs our output
        self.linear2 = nn.Linear(self.hidden_size, self.output_size, bias = False)

    def forward(self, x):
        #######################
        # TODO: Implement the forward pass
        #######################
        
        raise NotImplementedError("The forward pass is not yet implemented")

def train(model, train_features, train_labels, test_features, test_labels, 
                num_epochs=50, learning_rate=0.001):
    """
    Train the neural network model
    
    Args:
        model: The neural network model
        train_features: training features represented by token indices (tensor)
        train_labels: train labels(tensor)
        test_features: test features represented by token indices (tensor)
        test_labels: test labels (tensor)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        returns are optional, you could return training history with every N epoches and losses if you want
    """
    ######################## 
    # TODO: Implement the training loop
    # Hint:
    #   1. Use Adam as your optimizer (available in the optim.Adam() class) rather than SGD
    #######################

    raise NotImplementedError("The train function is not yet implemented")

def evaluate(model, test_features, test_labels):
    """
    Evaluate the trained model on test data
    
    Args:
        model: The trained neural network model
        test_features: (tensor)
        test_labels: (tensor)
    
    Returns:
        a dictionary of evaluation metrics (include test accuracy at the minimum)
        (You could import scikit-learn's metrics implementation to calculate other metrics if you want)
    """
    
    ####################### 
    # TODO: Implement the evaluation function
    # Hints: 
    # 1. Use torch.no_grad() for evaluation
    # 2. Use torch.argmax() to get predicted classes
    #######################
    
    return {
        'test_accuracy': 0.0, 
        'test_precision': 0.0,
        'test_recall': 0.0,
        'test_f1': 0.0,
    } # modify this return statement as you want


if __name__ == "__main__":
    
    ####################
    # TODO: If applicable, modify anything below this line 
    # according to your model configuration 
    # and to suit your need (naming changes, parameter changes, 
    # additional statements and/or functions)
    ####################

    # Load training and test data
    train_texts, train_labels = load_data('../data/train.txt')
    
    test_texts, test_labels = load_data('../data/test.txt')

    # Preprocess text
    processor = TextProcessor(vocab_size=10000)
    processor.build_vocab(train_texts) 
        
    # Convert text documents to tensor representations of word indices
    max_length = 100
    train_features = convert_text_to_tensors(train_texts, processor, max_length)
    test_features = convert_text_to_tensors(test_texts, processor, max_length)
    
    # Create a neural network model 
    # Modify the hyperparameters according to your model architecture
    vocab_size = len(processor.word_to_idx)
    embedding_dim = 100
    hidden_size = 64
    output_size = 2  # Binary classification for sentiment analysis
    
    model = NeuralNetwork(vocab_size, embedding_dim, hidden_size, output_size, max_length)
    
    # Train
    training_history = train(model, train_features, train_labels, test_features, test_labels, 
                                  num_epochs=50, learning_rate=0.001)
    
    # Evaluate
    evaluation_results = evaluate(model, test_features, test_labels)
    
    print(f"Model performance report: \n")
    print(f"Test accuracy: {evaluation_results['test_accuracy']:.4f}")
    print(f"Test F1 score: {evaluation_results['test_f1']:.4f}")

    # Save model weights to file
    outfile = '../trained_model.pth'
    torch.save(model.state_dict(), outfile)
    print(f"Trained model saved to {outfile}")
    
