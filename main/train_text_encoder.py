import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import itertools
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from collections import Counter
import os
import time
import re

import clip_classification
import training_utility


# ---------------------
# Section: Matrix-type Dataset / loss Preparation
# ---------------------
def prepare_sim_matrix_dataset(classname_list, synset_list, batch_size=10, global_sampling=False):
    similarity_matrix = np.zeros((len(classname_list), len(classname_list)))
    word_index = {word: i for i, word in enumerate(classname_list)}

    for i, synset1 in enumerate(synset_list):
        for j, synset2 in enumerate(synset_list):
            similarity_matrix[i, j] = synset1.wup_similarity(synset2)

    if global_sampling:
        # Create DataLoader for the batch training
        train_dataset_ = DataLoader(list(itertools.combinations(classname_list, 2)), batch_size=batch_size,
                                    shuffle=True, collate_fn=lambda batch: batch)  # This returns a list of tuples.
        print("Global sampling mode")
    else:
        train_dataset_ = DataLoader(classname_list, batch_size=batch_size, shuffle=True)
    return similarity_matrix, word_index, train_dataset_


class GraphDistanceLoss_sim_matrix(nn.Module):
    def __init__(self, wordnet_similarity_matrix, word_index, global_sampling=False, original_embeddings=None, reg_lambda=0.1, loss_type="square", loss_amp_const=4):
        """
        Args:
            wordnet_similarity_matrix: Precomputed similarity matrix.
            word_index: Mapping from word to index.
            global_sampling: If True, texts are provided as pairs (u, v).
            original_embeddings: Optional dict mapping words to their precomputed (e.g. CPU) original embeddings.
            reg_lambda: Regularization strength.
        """
        super(GraphDistanceLoss_sim_matrix, self).__init__()
        self.wordnet_similarity_matrix = wordnet_similarity_matrix
        self.word_index = word_index
        self.global_sampling = global_sampling
        self.original_embeddings = original_embeddings
        self.reg_lambda = reg_lambda
        self.loss_type = "square"
        self.loss_amp_const = loss_amp_const

    def forward(self, model_to_use, texts, tokenizer_):
        if self.global_sampling:
            # In global sampling mode, texts are already pairs of (u, v)
            unique_words = set()
            for pair in texts:
                unique_words.add(pair[0])
                unique_words.add(pair[1])

            # Encode each unique word once
            words_batch_list = list(unique_words)
            tokenized_inputs = tokenizer_(words_batch_list).to('cuda')

            # Encode the entire batch and normalize the embeddings
            curr_embedding = model_to_use.encode_text(tokenized_inputs)
            normalized_embeddings = F.normalize(curr_embedding, p=2, dim=-1)
            # Create a dictionary mapping each word to its embedding
            embeddings = {word: embedding for word, embedding in zip(words_batch_list, normalized_embeddings)}

            # Calculate loss for each pair
            loss = 0.0
            for pair in texts:
                u, v = pair[0], pair[1]
                i, j = self.word_index[u], self.word_index[v]
                target_similarity = self.wordnet_similarity_matrix[i, j]
                curr_loss = cosine_sim_loss(embeddings[u], embeddings[v], target_similarity, loss_type=self.loss_type, amplification_constant=self.loss_amp_const)
                loss += curr_loss

            # Normalize by the number of pairs
            loss /= len(texts)

            if self.original_embeddings is not None:
                # Stack current embeddings and original embeddings for all words in the batch
                curr_embeddings_batch = torch.stack(
                    [embeddings[word] for word in words_batch_list])  # Shape: (batch_size, embedding_dim)

                # Move original embeddings to the same device as current embeddings
                orig_embeddings_batch = torch.stack([self.original_embeddings[word] for word in words_batch_list]) # Shape: (batch_size, embedding_dim)

                # Compute the MSE loss for the entire batch, sum the losses
                reg_loss = F.mse_loss(curr_embeddings_batch, orig_embeddings_batch, reduction='sum') / len(words_batch_list)

                total_loss = loss + self.reg_lambda * reg_loss
                return total_loss, [loss, reg_loss * self.reg_lambda]
            
            return loss

        else:
            tokenized_inputs = tokenizer_(texts).to('cuda')
            # Encode the entire batch and normalize the embeddings
            curr_embedding = model_to_use.encode_text(tokenized_inputs)
            normalized_embeddings = F.normalize(curr_embedding, p=2, dim=-1)
            # Create a dictionary mapping each word to its embedding
            embeddings = {word: embedding for word, embedding in zip(texts, normalized_embeddings)}

            loss = 0.0
            all_combinations = list(itertools.combinations(texts, 2))

            for u, v in all_combinations:
                i, j = self.word_index[u], self.word_index[v]
                target_similarity = self.wordnet_similarity_matrix[i, j]
                curr_loss = cosine_sim_loss(embeddings[u], embeddings[v], target_similarity, loss_type=self.loss_type, amplification_constant=self.loss_amp_const)
                loss += curr_loss

            loss /= len(all_combinations)
            return loss


def cosine_sim_loss(embedding_1, embedding_2, target_similarity, loss_type="square", amplification_constant=4):
    current_similarity = F.cosine_similarity(embedding_1, embedding_2, dim=-1)
    similarity_diff = torch.abs(current_similarity - target_similarity) * amplification_constant
    if loss_type == "square":
        return similarity_diff ** 2
    elif loss_type == "logcosh":
        return torch.log(torch.cosh(similarity_diff))
    elif loss_type == "huber":
        delta = 1.0
        huber_loss = torch.where(similarity_diff <= delta,
                                 0.5 * (similarity_diff ** 2),  # Quadratic for small errors
                                 delta * (similarity_diff - 0.5 * delta))  # Linear for large errors
        return huber_loss
    elif loss_type == "exp":
        return torch.exp(similarity_diff) / 10
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")


# ---------------------
# Section: Training code
# ---------------------
# Helper function to extract class name and synset from a WordNet id.
def id_to_class_name_and_synset(wordnet_id):
    pos = wordnet_id[0]  # First letter indicates part of speech
    offset = int(wordnet_id[1:])  # Convert numeric part to integer
    curr_synset = wn.synset_from_pos_and_offset(pos, offset)
    return curr_synset.name().split('.')[0].replace("_", " ").replace("-", " "), curr_synset


def train_model_(model_to_tune, tokenizer_, train_loader,num_epochs, criterion,
                 scheduler=None, final_save_name='test',
                 use_custom_regularization=False):
    """
    Universal training function that supports an optional embedding regularization term.

    Args:
        model_to_tune: The model to be fine-tuned.
        tokenizer_: A function to tokenize input text.
        train_loader: DataLoader providing batches for training.
        num_epochs: Number of epochs to train.
        criterion: Loss function that takes (model, batch, tokenizer_) and returns a loss tensor.
        scheduler: (Optional) Learning rate scheduler.
        final_save_name: Filename for saving the final model checkpoint.
        use_custom_regularization: If True, applies an additional regularization term to preserve original embeddings.

    Returns:
        The fine-tuned model.
    """
    # --- Set up the model for training ---
    model_to_tune.train()
    named_parameters = list(model_to_tune.named_parameters())
    text_params = [p for n, p in named_parameters if 'visual' not in n]

    # Ensure gradients are enabled for text parameters.
    for param in text_params:
        param.requires_grad = True

    print(f"Number of text parameters: {len(text_params)}")

    # --- Choose optimizer based on the training mode ---
    if use_custom_regularization:
        optimizer = optim.SGD([{"params": text_params}], lr=1e-4, weight_decay=0)
    else:
        optimizer = optim.AdamW([{"params": text_params}], lr=1e-5)

    # --- Training loop ---
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        running_loss = 0.0
        running_reg_loss = 0.0

        # Iterate over training batches
        for batch in tqdm(train_loader, desc='Batches', leave=False):
            optimizer.zero_grad()

            loss_info = criterion(model_to_tune, batch, tokenizer_)
            if isinstance(loss_info, (tuple, list)):
                loss, loss_split = loss_info
            else:
                loss = loss_info
                loss_split = None

            if isinstance(loss, float) or not torch.is_tensor(loss):
                continue
            loss.backward()
            optimizer.step()
            running_loss += loss_split[0].item() if loss_split else loss.item()
            running_reg_loss += loss_split[1].item() if loss_split else 0

        avg_loss = running_loss / len(train_loader)
        avg_reg_loss = running_reg_loss / len(train_loader)
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, " f"Reg Loss: {avg_reg_loss:.4f} (lambda: {criterion.reg_lambda})")

        # Update learning rate if a scheduler is provided.
        if scheduler:
            scheduler.step()

    # --- Save the final model checkpoint ---
    checkpoint_dict = {
        "epoch": num_epochs,
        "state_dict": model_to_tune.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    target_dir = "../tuned_models"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    torch.save(checkpoint_dict, f"{target_dir}/{final_save_name}.pt")

    return model_to_tune


def lexical_training(
        model_info=('ViT-B-32', 'laion400m_e31'),
        training_type='hypernym',  # "synonym" or "hypernym"
        dataset_name="imagenet",
        target_file_name="100_hypernym_level_1",
        batch_size=10,
        number_of_epochs=10,
        save_name="test",
        global_sampling=False,
        custom_regularization=False,
        custom_lambda=0.1,
        loss_type="square",
        loss_amp_const=4,
):
    if training_type.lower() not in ["synonym", "hypernym", "mixed"]:
        print("Invalid training type, please choose 'synonym' or 'hypernym', exiting")
        exit(1)

    print("Function arguments and local variables:")
    for name, value in locals().items():
        print(f"{name}: {value}")

    # Initialize tokenizer and model (common setup).
    model_type = model_info[0]
    pretrained_dataset = model_info[1]
    tokenizer = open_clip.get_tokenizer(model_type)
    model_for_tune, _, _ = open_clip.create_model_and_transforms(model_type, pretrained=pretrained_dataset)
    model_for_tune = model_for_tune.to('cuda')

    # Load JSON data common to both routines.
    with open(f"../zero_shot_analyze/{dataset_name}/{target_file_name}.json", "r") as f:
        data = json.load(f)
        original_ids = data['original_id']
        related_ids = data['related_id']

    # Prepare testing subset for evaluation.
    index = training_utility.prepare_index_for_checking(related_ids)
    combined_array_names = []
    synset_list = []

    if training_type.lower() == "synonym":
        # Process original ids.
        for curr_id in original_ids:
            curr_class_name, curr_synset = id_to_class_name_and_synset(curr_id)
            combined_array_names.append(curr_class_name)
            synset_list.append(curr_synset)
        for i in range(len(original_ids)):
            combined_array_names += related_ids[i]
            for _ in range(len(related_ids[i])):
                synset_list.append(synset_list[i])

    elif training_type.lower() == "hypernym":

        # Process original ids.
        for curr_id in original_ids:
            curr_class_name, curr_synset = id_to_class_name_and_synset(curr_id)
            combined_array_names.append(curr_class_name)
            synset_list.append(curr_synset)
        # Process related ids.
        for id_list in related_ids:
            for curr_id in id_list:
                curr_class_name, curr_synset = id_to_class_name_and_synset(curr_id)
                combined_array_names.append(curr_class_name)
                synset_list.append(curr_synset)
                
    else: # training_type.lower() == "mixed"
        for curr_id in original_ids:
            curr_class_name, curr_synset = id_to_class_name_and_synset(curr_id)
            combined_array_names.append(curr_class_name)
            synset_list.append(curr_synset)
            # Process related ids.
        for i in range(len(original_ids)):
            for curr_id in related_ids[i]:
                if re.match(r'^[nva]\d+', curr_id):
                    curr_class_name, curr_synset = id_to_class_name_and_synset(curr_id)
                    combined_array_names.append(curr_class_name)
                    synset_list.append(curr_synset)
                else:
                    combined_array_names.append(curr_id)
                    synset_list.append(synset_list[i])

    synset_subset = [synset_list[i] for i in index]

    # print(combined_array_names)
    # Check for duplicates.
    if len(combined_array_names) != len(set(combined_array_names)):
        print("Duplicates found:", training_utility.find_duplicates(combined_array_names),
              ", please check the dataset, exiting")
        # print the location of the duplicates
        raise ValueError("Duplicates found, please check the dataset, exiting")
    else:
        print("No duplicates, continue training")

    sim_matrix, word_idx, dataset_for_training = prepare_sim_matrix_dataset(
        combined_array_names, synset_list, batch_size=batch_size, global_sampling=global_sampling
    )
    if custom_regularization:
        with torch.no_grad():
            print(
                f"Precomputing original embeddings for {len(combined_array_names)} unique words for custom Regularization.")
            # Compute and store the normalized embedding for each unique word.
            tokenized_inputs = tokenizer(combined_array_names).to('cuda')
            # Encode the entire batch and normalize the embeddings
            curr_embedding = model_for_tune.encode_text(tokenized_inputs)
            normalized_embeddings = F.normalize(curr_embedding, p=2, dim=-1)
            # Create a dictionary mapping each word to its embedding
            original_embeddings = {word: embedding.detach() for word, embedding in
                                   zip(combined_array_names, normalized_embeddings)}

    else:
        original_embeddings = None
    loss_for_training = GraphDistanceLoss_sim_matrix(
        sim_matrix,
        word_idx,
        global_sampling=global_sampling,
        original_embeddings=original_embeddings,
        reg_lambda=custom_lambda,
        loss_type=loss_type,
        loss_amp_const=loss_amp_const
    )

    testing_subset = [combined_array_names[i] for i in index]

    # Record distances before fine-tuning.
    with torch.no_grad():
        before_finetune = training_utility.store_distance_2d_array(model_for_tune, tokenizer, testing_subset)

    # Define a save name and fine-tune the model.
    final_save_name = f"{save_name}_{dataset_name}_{global_sampling}_{target_file_name}_{custom_regularization}_{training_type}_{loss_type}"

    model_trained_ = train_model_(
        model_to_tune=model_for_tune,
        tokenizer_=tokenizer,
        train_loader=dataset_for_training,
        num_epochs=number_of_epochs,
        criterion=loss_for_training,
        scheduler=None,
        final_save_name=final_save_name,
        use_custom_regularization=custom_regularization,
    )

    target_distance = training_utility.get_target_similarity_matrix_(testing_subset, synset_subset)
    with torch.no_grad():
        after_finetune = training_utility.store_distance_2d_array(model_trained_, tokenizer, testing_subset)

    error_stats = training_utility.visualize_improvement_matrix(before_finetune, after_finetune, target_distance,
                                                                testing_subset)
    return final_save_name, error_stats


if __name__ == "__main__":
    start_time = time.time()

    dataset_name = "imagenet"
    target_file_name = "100_synonym_sample"
    if "hypernym" in target_file_name:
        training_type = "hypernym"
    elif "synonym" in target_file_name:
        training_type = "synonym"
    else:
        training_type = "mixed"

    model_info = ("ViT-B-32", "laion400m_e31")

    tuned_model_name, _ = lexical_training(
        model_info=model_info,
        training_type=training_type,
        dataset_name=dataset_name,
        target_file_name=target_file_name,
        batch_size=10,
        number_of_epochs=7,
        save_name="test",
        global_sampling=True,
        custom_regularization=True,
        custom_lambda=0.08,
        loss_type="square",
        loss_amp_const=3,
    )

    end_time = time.time()  # End timer
    print("Training time: {:.2f} seconds".format(end_time - start_time))

    start_time = time.time()

    clip_classification.run_classification(
        model_info=model_info,
        dataset_name=dataset_name,
        include_original_name=False,  # False
        target_file_name=target_file_name,
        tuned_model_save_name=tuned_model_name,
        combination_sampled=1 if training_type == "hypernym" else 3,
        template_type="null"
    )
    #
    # clip_classification.run_classification(
    #     model_info=model_info,
    #     dataset_name=dataset_name,
    #     include_original_name=False,  # False
    #     target_file_name=target_file_name,
    #     tuned_model_save_name=tuned_model_name,
    #     combination_sampled=1 if training_type == "hypernym" else 3,
    #     template_type="general"
    # )

    end_time = time.time()  # End timer

    print("Inference testing time: {:.2f} seconds".format(end_time - start_time))
