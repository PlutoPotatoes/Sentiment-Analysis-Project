import torch
import re
from collections import Counter

#########################################################
# COMP331 Fall 2025 PA2
# This file contains utilities for loading the datasets and text preprocessing
# The TextProcessor class contains methods similar to those in the tutorial
#########################################################

def load_data(infile):
    """
    Load data from a text file with the data format: <text>\t<label>
    """
    texts = []
    labels = []

    with open(infile, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # There could be tabs in text, so we use the rightmost tab to separate text and label
            last_tab = line.rfind('\t')
            if last_tab != -1: # if a tab is found
                text = line[:last_tab]
                label = line[last_tab+1:]
                texts.append(text)
                labels.append(int(label))
    return texts, torch.tensor(labels, dtype=torch.long)


class TextProcessor:
    """
    Preprocess text, build vocabulary and mappings between words and indices
    """
    def __init__(self, vocab_size=5000):
        # We restrict vocabulary size to a pre-defined value to save memory
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def tokenize(self, text):
        """
        Simple tokenization, remove punctuation and convert to lowercase
        You can try other tokenization methods, such as NLTK, spaCy, etc.
        """
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()
    
    def build_vocab(self, texts):
        """
        Build vocabulary, mappings between words and indices from training corpus
        """
        word_counts = Counter()
        
        for text in texts:
            words = self.tokenize(text)
            word_counts.update(words)
        
        # Build vocabulary based on the most common words, constrained by vocabulary size
        # Two reserve spaces for <PAD> and <UNK>
        most_common = word_counts.most_common(self.vocab_size - 2) 
        
        # Create word to index mapping
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for i, (word, count) in enumerate(most_common):
            self.word_to_idx[word] = i + 2
            self.idx_to_word[i + 2] = word
    
    def pad_sequence(self, sequence, max_length=25):
        """
        Truncate a sequence if too long (> max_length), pad if too short (< max_length)
        """
        if len(sequence) < max_length:
            return sequence + ["<PAD>"] * (max_length - len(sequence))
        return sequence[:max_length]

    def text_to_indices(self, text, max_length=25):
        """
        Convert text to sequence of word indices. 
        Pad/Truncate the sequence to max_length. 
        """
        words = self.tokenize(text)
        words = self.pad_sequence(words, max_length)

        # Convert to indices after tokenization and padding
        indices = [self.word_to_idx.get(token, self.word_to_idx["<UNK>"]) for token in words]
            
        return torch.tensor(indices, dtype=torch.long)
    
    def get_vocab_size(self):
        """
        Return the vocabulary size
        """
        return len(self.word_to_idx)
    
    def get_word(self, idx):
        """
        Access word by index
        """
        return self.idx_to_word.get(idx, '<UNK>')
    
    def get_idx(self, word):
        """
        Access index by word
        """
        return self.word_to_idx.get(word, self.word_to_idx['<UNK>'])

def convert_text_to_tensors(docs, processor, max_length=100):
    """
    This function prepares training features given a text corpus and a TextProcessor instance. 
    It converts raw texts into tensor representations of word indices.

    Args:
        docs: List of raw text strings
        processor: a TextProcessor instance with built vocabulary
        max_length: Maximum sequence length

    Returns:
        Tensor of shape (num_texts, max_length)
    """
    token_indices = []

    for i, text in enumerate(docs):
        indices = processor.text_to_indices(text, max_length)
        token_indices.append(indices)
    
    token_indices = torch.stack(token_indices)
    
    return token_indices
