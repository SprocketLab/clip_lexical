import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import wordnet as wn
import torch.nn.functional as F


"""
This file contains utility functions for evaluating the performance of the fine-tuned model base on whether it fulfill our need.
"""
def find_duplicates(arr):
    counts = Counter(arr)
    duplicates = [item for item, count in counts.items() if count > 1]
    return duplicates


def prepare_index_for_checking(related_id_array):
    index = []
    buffer = 0
    class_count = len(related_id_array)
    for i in range(5):
        index.append(i)
        index.append(i + buffer + class_count)
        buffer = buffer + len(related_id_array[i])-1
    return index


def store_distance_2d_array(model_to_check, tokenizer_, texts_list_):
    tokenized_inputs = tokenizer_(texts_list_).to('cuda')
    # Encode the entire batch and normalize the embeddings
    curr_embedding = model_to_check.encode_text(tokenized_inputs)
    embeddings = F.normalize(curr_embedding, p=2, dim=-1)
    cosine_similarity_matrix = np.zeros((len(texts_list_), len(texts_list_)))
    for i in range(len(texts_list_)):
        for j in range(len(texts_list_)):
            cosine_similarity = F.cosine_similarity(embeddings[i], embeddings[j], dim=-1)
            cosine_similarity_matrix[i, j] = cosine_similarity.detach().cpu().item()
    return cosine_similarity_matrix


def get_target_similarity_matrix_(word_list, synset_list=None, synonym_hps=None):
    if synset_list is None:
        # we adapt the synonym method to get the target matrix
        synonym_boundary = synonym_hps[0]
        unrelated_boundary = synonym_hps[1]
        antonym_boundary = synonym_hps[2]
        matrix = np.full((len(word_list), len(word_list)), unrelated_boundary)
        np.fill_diagonal(matrix, 1.0)
        for i in range(len(word_list) - 1):
            matrix[i, i + 1] = synonym_boundary
            matrix[i + 1, i] = synonym_boundary
        return matrix

    similarity_matrix = np.zeros((len(word_list), len(word_list)))
    for i, synset1 in enumerate(synset_list):
        for j, synset2 in enumerate(synset_list):
            similarity_matrix[i, j] = synset1.wup_similarity(synset2)
    return similarity_matrix


def visualize_improvement_matrix(before_matrix, after_matrix, target_matrix, test_labels):
    """
    Visualizes improvement matrix using pandas DataFrames with formatted strings.

    Args:
        before_matrix (2D array): Distance matrix before fine-tuning
        after_matrix (2D array): Distance matrix after fine-tuning
        target_matrix (2D array): Target distance matrix
        test_labels (list): Labels for matrix rows/columns
    """
    # Create formatted matrix
    error_before, error_after, squared_error_before, squared_error_after = 0.0, 0.0, 0.0, 0.0
    formatted_matrix = []
    for i in range(len(test_labels)):
        row = []
        for j in range(len(test_labels)):
            # Extract values and format
            a = float(before_matrix[i][j])
            b = float(after_matrix[i][j])
            c = float(target_matrix[i][j])
            error_before += abs(a - c)
            error_after += abs(b - c)
            squared_error_before += (a - c) ** 2
            squared_error_after += (b - c) ** 2
            row.append(f"{a:.2f}->{b:.2f} ({c:.2f})")
        formatted_matrix.append(row)

    # Create DataFrame with labels
    df = pd.DataFrame(formatted_matrix,
                      index=test_labels,
                      columns=test_labels)

    # Improve readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_rows', None)

    print("Formatted Distance Matrix (Before->After (Target)):")
    print(df)
    print("Error Metric inside the subset")
    print(f"Total error before fine-tuning: {error_before:.2f}, Total error after fine-tuning: {error_after:.2f}\n"
          f"Total squared error before fine-tuning: {squared_error_before:.2f}, Total squared error after fine-tuning: {squared_error_after:.2f}")

    return [error_before, error_after, squared_error_before, squared_error_after]