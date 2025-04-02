import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
from open_clip import load_checkpoint
import copy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import itertools
import nltk
from nltk.corpus import wordnet as wn
import random
import json
import re


class custom_classification(Dataset):
    def __init__(self, path_csv, transform=None):
        self.data = pd.read_csv(path_csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = Image.open(img_path)
        label = self.data.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        return image, label


def run_CLIP(model_, tokenizer_, class_names_, class_templates_, torch_dataset_):
    loader = torch.utils.data.DataLoader(torch_dataset_, batch_size=500)
    texts = [template.format(cls) for template in class_templates_ for cls in class_names_]
    print('Texts:', texts)
    tokenize_texts = tokenizer_(texts).to('cuda')
    with torch.no_grad():
        text_features = model_.encode_text(tokenize_texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    top1, n = 0., 0.

    model_.eval()
    with torch.no_grad():
        for images, target in loader:
            images = images.to('cuda')
            target = target.to('cuda')
            # predict
            image_features = model_.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # stack text features with batch size
            logits = 100. * image_features @ text_features.t()

            probability = torch.nn.functional.softmax(logits, dim=-1)
            prediction = probability.argmax(dim=-1)
            top1 += (prediction == target).sum().item()
            n += images.size(0)

    top1 = top1 / n * 100
    print(f"Top-1 accuracy: {top1:.2f}%\n")
    del tokenize_texts, text_features, logits, probability, prediction, image_features
    torch.cuda.empty_cache()
    return top1


def synonym_comparison(model_, tokenizer_, original, combination_list, class_templates_, torch_dataset_):
    print("Length of dataset:", len(torch_dataset_))
    accuracy_original = run_CLIP(model_, tokenizer_, original, class_templates_, torch_dataset_)
    avg_accuracy = 0
    with torch.no_grad():
        for this_comb in combination_list:
            accuracy = run_CLIP(model_, tokenizer_, this_comb, class_templates_, torch_dataset_)

            # free cuda memory
            torch.cuda.empty_cache()

            avg_accuracy += accuracy

    avg_accuracy /= len(combination_list)
    return accuracy_original, avg_accuracy


def load_model(path=None, model_type='ViT-B-32'):
    clip_model, _, preprocess_ = open_clip.create_model_and_transforms(model_type)
    load_checkpoint(clip_model, path)
    clip_model.to('cuda')
    return clip_model


def generate_combination(synonym_list_, count=50):
    result = []
    for i in range(count):
        temp = []
        for syn in synonym_list_:
            temp.append(random.choice(syn))
        result.append(temp)
    return result


def to_class_name(input_string):
    if re.match(r'^[nva]\d+', input_string):
        # Try to check if it's a valid synset in WordNet
        try:
            pos = input_string[0]  # First letter indicates part of speech
            offset = int(input_string[1:])  # Convert numeric part to integer
            curr_synset = wn.synset_from_pos_and_offset(pos, offset)
            return curr_synset.name().split('.')[0].replace('_', ' ').replace('-', ' ')
        except:
            return input_string
    return input_string


def run_classification(model_info=("ViT-B-32", "laion400m_e31"), dataset_name="imagenet", target_file_name="",
                       tuned_model_save_name="hypernym_test", include_original_name=True, combination_sampled=1, template_type="null"):
    with open(f"../zero_shot_analyze/{dataset_name}/{target_file_name}.json", "r") as f:
        data = json.load(f)
        original_id = data['original_id']
        related_id = data['related_id']
    original_names = []
    related_lists = []
    for curr_id in original_id:
        original_names.append(to_class_name(curr_id))

    for id_list in related_id:
        related_list = []
        for curr_id in id_list:
            related_list.append(to_class_name(curr_id))
        related_lists.append(related_list)

    template_face = ['a face that looks {}']
    template_null = ['{}']
    template_general = ['a photo of {}']
    if template_type == "null":
        template = template_null
    elif template_type == "general":
        template = template_general
    elif template_type == "face":
        template = template_face

    if include_original_name:
        print("Including original names in the class_names")
        for i, related_list_ in enumerate(related_lists):
            related_list_.append(original_names[i])

    model_type = model_info[0]
    pretrained_dataset = model_info[1]
    model, _, preprocess = open_clip.create_model_and_transforms(model_type, pretrained=pretrained_dataset)
    model.eval()
    model = model.to('cuda')

    tokenizer = open_clip.get_tokenizer(model_type)

    random_name_replacement = generate_combination(related_lists, count=combination_sampled)

    dataset_to_use = custom_classification(f'../zero_shot_analyze/{dataset_name}/{target_file_name}.csv', transform=preprocess)

    original_accuracy, original_accuracy_replacement = synonym_comparison(model, tokenizer, original_names,
                                                                          random_name_replacement,
                                                                          template, dataset_to_use)

    del model

    tuned_model = load_model(path=f'../tuned_models/{tuned_model_save_name}.pt', model_type=model_type)
    tuned_model.eval()
    tuned_accuracy, tuned_accuracy_replacement = synonym_comparison(tuned_model, tokenizer, original_names,
                                                                    random_name_replacement,
                                                                    template, dataset_to_use)

    original_improvement = tuned_accuracy - original_accuracy
    replaced_improvement = tuned_accuracy_replacement - original_accuracy_replacement

    print(f"Original name improvement after finetuning, with {template_type} template: {original_improvement:.2f}% ({original_accuracy:.2f} → {tuned_accuracy:.2f})")
    print(f"Replaced name improvement after finetuning, with {template_type} template: {replaced_improvement:.2f}% ({original_accuracy_replacement:.2f} → {tuned_accuracy_replacement:.2f})")
    return original_accuracy, tuned_accuracy, original_accuracy_replacement, tuned_accuracy_replacement


if __name__ == "__main__":
    run_classification(
        model_info=("ViT-B-32", "laion400m_e31"),
        dataset_name="imagenet",
        include_original_name=False,
        target_file_name="100_synonym_1",
        tuned_model_save_name="test_imagenet_True_100_synonym_1_True_synonym_square",
        combination_sampled=2,
        template_type="general"
    )
