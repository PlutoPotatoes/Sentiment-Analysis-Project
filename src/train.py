import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import load_data, TextProcessor, convert_text_to_tensors
from torch.utils.data import TensorDataset, DataLoader

import random

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
        # Define your model class (Done - Ryan)
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

        #add any other layers here
        #other layers should follow the same linear-activation-linear sandwich strategy

        #final hidden layer, takes hidden in outputs our output
        self.output_layer = nn.Linear(self.hidden_size, self.output_size, bias = False)

    def forward(self, x):
        #######################
        # Implement the forward pass (Done? - Ryan)
        #######################
        embed = self.embedding(x) #[batch_size, sequence_len, embed_dim] sized matrix
        hidden1 = self.linear1(embed) #[batch_size, sequence_len, hidden_size]
        relu = self.relu(hidden1)

        #implement any other layers here

        #now we pool and output our data and pass to our final layer
        mean = torch.mean(relu, dim=1)
        res = self.output_layer(mean)
        return res



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

    batch = 64
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)   

    train_dataset = TensorDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    train_history = []
    #set model to train mode?
    model.train()

    for j, epoch in enumerate(range(num_epochs)):
        total_loss = 0
        for i, (features, targets) in enumerate(train_dataloader):
            #reset optimizer
            optimizer.zero_grad()
            #make our predictions
            predictions = model(features)
            #check how the model did and calculate loss
            losses = loss_function(predictions, targets)
            losses.backward()
            #move our optimizer forwards one step
            optimizer.step()
            #tally losses
            total_loss += losses.item()

        #print results of 100 epochs
        if j % 10 == 0:  # Print only every 100 epochs
            avg_loss = total_loss / len(train_dataloader)
            train_history.append(avg_loss)
            print(f"Epoch {epoch}: average training loss: {avg_loss}")

    return train_history

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
    
    model.eval()

    with torch.no_grad():
        prediction = model(test_features)
        predicted_class = torch.argmax(prediction, dim=1)
        #check and add to f1 calculations

    TP = 0
    FP= 0
    TN = 0
    FN = 0
    total = predicted_class.size(dim=0)
    for i in range(predicted_class.size(dim=0)):
        if predicted_class[i] == test_labels[i]:
            if predicted_class[i] == 1:
                TP+=1
            else:
                TN+=1
        else:
            if predicted_class[i] == 1:
                FP+=1
            else:
                FN+=1



    return {
        'test_accuracy': (TP+TN)/total, 
        'test_precision': TP/(TP+FP),
        'test_recall': TP/(TP+FN),
        'test_f1': (2*(TP/(TP+FP))*(TP/(TP+FN)))/(TP/(TP+FP) + (TP/(TP+FN))),
    }


if __name__ == "__main__":
    
    ####################
    # TODO: If applicable, modify anything below this line 
    # according to your model configuration 
    # and to suit your need (naming changes, parameter changes, 
    # additional statements and/or functions)
    ####################

    # Load training and test data
    train_texts, train_labels = load_data('src/data/train.txt')
    
    test_texts, test_labels = load_data('src/data/test.txt')

#=====================================================
    #combine test and train data for shuffling
    all_texts = train_texts + test_texts
    all_labels = torch.cat((train_labels, test_labels))


    # zip together for shuffling
    combined_data = list(zip(all_texts, all_labels.tolist()))
    
    # shuffle
    random.shuffle(combined_data)
    
    # Unzip
    shuffled_texts, shuffled_labels_list = zip(*combined_data)
    
    # Convert back to a tensor
    shuffled_labels = torch.tensor(shuffled_labels_list, dtype=torch.long)

    # split 80% for train, 20% for test
    split_idx = int(len(shuffled_texts) * 0.8) # 80% for train, 20% for test
    
    train_texts = shuffled_texts[:split_idx]
    train_labels = shuffled_labels[:split_idx]
    
    test_texts = shuffled_texts[split_idx:]
    test_labels = shuffled_labels[split_idx:]
    
    #=====================================================


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
                                  num_epochs=30, learning_rate=0.001)
    
    print(training_history)

    # Evaluate
    evaluation_results = evaluate(model, test_features, test_labels)
    
    print(f"Model performance report: \n")
    print(f"Test accuracy: {evaluation_results['test_accuracy']:.4f}")
    print(f"Test F1 score: {evaluation_results['test_f1']:.4f}")
    print(f"Test Precision score: {evaluation_results['test_precision']:.4f}")
    print(f"Test Recall score: {evaluation_results['test_recall']:.4f}")

    # Save model weights to file
    outfile = 'src/models/trained_model.pth'
    torch.save(model.state_dict(), outfile)
    print(f"Trained model saved to {outfile}")
    
