import torch 
import torch.nn as nn
from utils import load_data, TextProcessor, convert_text_to_tensors
from train import NeuralNetwork
from torch.utils.data import TensorDataset, DataLoader

#########################################################
# COMP331 Fall 2025 PA2
# This file contains functions to load a pretrained model   
# and make predictions on unlabeled data for you to implement
#########################################################


def load_model(model_path, vocab_size, embedding_dim, hidden_size, output_size, max_length=100):
    """
    Load a pre-trained model from file
    
    Args:
        model_path: Path to saved model file
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_size: Size of hidden layer
        output_size: Number of output classes
        max_length: Maximum sequence length
        
        (Change arguments according to your model architecture if needed)
    
    Returns:
        Loaded model in evaluation mode
    """
    ################################
    # TODO: Create a model instance and load state dict
    # Hints: 
    #   1. Create model: model = NeuralNetwork(vocab_size, embedding_dim, hidden_size, output_size, max_length)
    #   2. Load weights: model.load_state_dict(torch.load(model_path))
    ################################   
    model = NeuralNetwork(vocab_size, embedding_dim, hidden_size, output_size, max_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_unlabeled_data(model, processor, unlabeled_texts, outfile, batch_size=1000, max_length=100):
    """
    Make predictions for unlabeled text documents and saves them to a file.
    
    Args:
        model: Pre-trained neural network model for IMDB sentiment classification
        processor: a TextProcessor instance
        unlabeled_texts: List of unlabeled text strings
        outfile: Path to save predictions
        batch_size: Batch size for processing
        max_length: Maximum sequence length
    """
    ########################################################
    # TODO: Implement prediction on unlabeled data
    # Hints: 
    #  1. Use torch.no_grad() for inference
    #  2. Process texts in batches for memory efficiency
    #  3. Save predictions to file in the same format as training data: 
    #       "<text>\t<predicted_label>"
    #  4. Decompose it into smaller tasks if you want
    ########################################################   

    #TODO YOU ARE HERE, FIGURE THIS SHIT OUT - Ryan
    train_dataloader = DataLoader(unlabeled_texts, batch_size=batch_size)
    for i, (features, targets) in enumerate(train_dataloader):
        predictions = model(features)
        print(predictions)





if __name__ == "__main__":

    
    # Load the unlabeled data (Labels were set to -1)
    unlabeled_file = 'src/data/unlabeled.txt'

    print("Predicting labels for unlabeled data")

    # Labels (set to -1) are not used for unlabeled data
    unlabeled_texts, _ = load_data(unlabeled_file)  

    # Load training data to build vocabulary (required for text preprocessing)
    train_file = 'src/data/train.txt'
    train_texts, _ = load_data(train_file)

    ####################
    #TODO: Modify anything below this line based on your model and for your need 
    # (if applicable)
    ####################

    # Preprocess text and build vocabulary 
    # vocabulary size and max length must match the trained model
    vocab_size = 10000
    max_length = 100
    processor = TextProcessor(vocab_size=vocab_size)
    processor.build_vocab(train_texts)
    # Define model parameters (must match the trained model)
    embedding_dim = 100
    hidden_size = 64
    output_size = 2

    # Load the trained model
    model_path = 'src/models/trained_model.pth'

    try:
        model = load_model(model_path, vocab_size, embedding_dim, hidden_size, output_size, max_length)
    except FileNotFoundError:
        print(f"Trained model file '{model_path}' not found. Run train.py first.")
        exit()
    
    # Predict labels and save to file
    outfile = 'src/data/predictions.txt'
    predict_unlabeled_data(model, processor, unlabeled_texts, outfile, max_length=max_length)
