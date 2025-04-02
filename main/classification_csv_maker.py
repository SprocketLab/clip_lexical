import os
import pandas as pd

from tqdm import tqdm
import random
from nltk.corpus import wordnet as wn
import json
import yaml


with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
IMAGENET_ROOT_PATH = config["imagenet_root_path_train"]


# ---------------------
# Section: utility code for building CSV files
# ---------------------
def get_imagenet_classes(hypernym_only=False, synonyms_only=False, have_both=False):
    f = open('../dataset_management/dir_label_name_1k.json')
    map_collection = json.load(f)
    f.close()
    class_id_list = []
    for key, values in map_collection.items():
        class_id_list.append(values[0])
    all_synset = []
    for curr_id in class_id_list:
        pos = curr_id[0]  # First letter indicates part of speech
        offset = int(curr_id[1:])  # Convert numeric part to integer
        synset = wn.synset_from_pos_and_offset(pos, offset)

        hypernyms = synset.hypernyms()
        synonyms = synset.lemmas()

        if hypernym_only:
            if len(hypernyms) != 0:
                all_synset.append(synset)
        elif synonyms_only:
            if len(synonyms) > 1:
                all_synset.append(synset)
        elif have_both:
            if len(hypernyms) != 0 and len(synonyms) > 1:
                all_synset.append(synset)
        else:
            all_synset.append(synset)
    # print(f"Total classes: {len(all_synset)}")
    return all_synset


def find_all_synonyms(synset, original_name):
    synonyms = []
    for lemma_name in synset.lemma_names():
        lemma_name = lemma_name.replace("_", " ").replace("-", " ")
        if lemma_name != original_name:
            synonyms.append(lemma_name)
    return list(set(synonyms))


def find_all_hypernyms(synset, level):
    if level < 1:
        return []
    direct = synset.hypernyms()
    all_hypernyms = list(direct)
    if level > 1:
        for hyper in direct:
            all_hypernyms.extend(find_all_hypernyms(hyper, level - 1))
    # remove duplicates
    all_hypernyms = list(set(all_hypernyms))
    return all_hypernyms


def get_corresponding_name(input_object):
    # Check if input_object is already a Synset
    if hasattr(input_object, 'name'):
        # Assume it's a Synset and process it directly
        name = input_object.name().split(".")[0].replace("_", " ").replace("-", " ")
        return name
    try:
        # Otherwise, assume it's a string like 'n12345'
        synset_corr = wn.synset_from_pos_and_offset(input_object[0], int(input_object[1:]))
        to_return = synset_corr.name().split(".")[0].replace("_", " ").replace("-", " ")
        return to_return
    except Exception as e:
        return input_object


def find_map_without_duplicates(input_map, random_subset=None):
    used_names = set()  # will hold all get_corresponding_name() results we have seen
    filtered_map = {}  # our result dictionary

    # Iterate over each key-value pair in the input map
    for key, value_list in input_map.items():
        # Get the string representation of the key
        key_name = get_corresponding_name(key)
        if key_name in used_names:
            # Already used; skip this entry
            continue

        # Check each element in the list for duplicates
        duplicate_found = False
        for element in value_list:
            element_name = get_corresponding_name(element)
            if element_name in used_names:
                duplicate_found = True
                break  # no need to check further; skip this key-value pair

        if duplicate_found:
            continue  # skip adding this entry

        # If we reach here, neither the key's name nor any of its value names
        # are in used_names. So we add them.
        filtered_map[key] = value_list
        used_names.add(key_name)
        for element in value_list:
            used_names.add(get_corresponding_name(element))
            
    # If a random subset size is specified, sample that many keys from filtered_map
    if random_subset is not None:
        keys = list(filtered_map.keys())
        # Ensure we don't sample more keys than are available
        sample_size = min(random_subset, len(keys))
        selected_keys = random.sample(keys, sample_size)
        filtered_map = {k: filtered_map[k] for k in selected_keys}
    # debug_names = []
    # for key, value_list in filtered_map.items():
    #     debug_names.append(get_corresponding_name(key))
    #     for element in value_list:
    #         debug_names.append(get_corresponding_name(element))
    #
    # print("Debugging names list:", debug_names)
    return filtered_map


# ---------------------
# Section: dataset_task specific code for generating CSV / json file pairs
# ---------------------
def fer_csv_synonym(image_to_test=100):
    root_path = f"{config['fer_2013_root_path']}/test/"
    emotion_list = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    emotion_synonym_list = [
        ['furious', 'wild', 'raging', 'tempestuous'],
        ['nauseate', 'churn up', 'sicken', 'gross out', 'revolt', 'repel'],
        ['fearfulness', 'reverence', 'fright', 'dread', 'awe'],
        ['glad', 'felicitous'],
        ['indifferent'],
        ['deplorable', 'distressing', 'lamentable', 'sorry', 'pitiful'],
        ["astonished", "amazed", "shocked"]
    ]
    image_list = []
    for i, emotion in enumerate(emotion_list):
        emotion_path = root_path + emotion
        files = random.sample(os.listdir(emotion_path), image_to_test)
        for file in files:
            # select 100 files for each emotion
            if file.endswith(".jpg"):
                image_path = emotion_path + "/" + file
                image_list.append({"label": i, "filepath": image_path})

    file_name = "../zero_shot_analyze/fer2013/fer2013_synonym"
    df = pd.DataFrame(image_list)
    df.to_csv(f"{file_name}.csv", index=False)

    dict_to_store = {
        "original_id": ["a00113818", "n07503260", "n07519253", "a01148283", "a01818992", "a01361863", "n07510348"],
        "related_id": emotion_synonym_list
    }

    with open(f"{file_name}.json", 'w') as f_:
        json.dump(dict_to_store, f_, indent=4)


def build_imagenet_csv(wordnet_id_list, image_to_test=30, csv_name="imagenet.csv"):
    image_list = []
    for i, name in enumerate(wordnet_id_list):
        class_path = IMAGENET_ROOT_PATH + name
        random_files = random.sample(os.listdir(class_path), image_to_test)
        for file in random_files:
            if file.endswith(".JPEG"):
                image_path = class_path + "/" + file
                image_list.append({"label": i, "filepath": image_path})

    df = pd.DataFrame(image_list)
    print("Number of images:", len(df))
    df.to_csv(csv_name, index=False)


def imagenet_csv_synonym(number_of_classes=300, random_id=1):
    synonym_map = {}
    wordnet_id_map = {}
    imagenet_classes = get_imagenet_classes(synonyms_only=True)
    for synset in imagenet_classes:
        original_name = synset.name().split(".")[0].replace("_", " ").replace("-", " ")
        wordnet_id_map[original_name] = f"{synset.pos()}{synset.offset():08d}"
        synonyms = find_all_synonyms(synset, original_name)
        if not synonyms:
            continue
        synonym_map[original_name] = synonyms

    filtered_classes = find_map_without_duplicates(synonym_map, random_subset=number_of_classes)
    print("Number of classes:", len(filtered_classes))
    # only keep those which is in filtered_classes
    wordnet_id_map = {key: wordnet_id_map[key] for key in filtered_classes.keys()}

    file_name = f'../zero_shot_analyze/imagenet/{len(filtered_classes)}_synonym_{random_id}'
    wordnet_id_list = list(wordnet_id_map.values())
    dict_to_store = {
        "original_id": wordnet_id_list,
        "related_id": list(filtered_classes.values())
    }

    with open(f"{file_name}.json", 'w') as f_:
        json.dump(dict_to_store, f_, indent=4)

    build_imagenet_csv(wordnet_id_list, image_to_test=30, csv_name=f"{file_name}.csv")

    return len(filtered_classes), file_name.split("/")[-1]


def imagenet_csv_hypernym(number_of_classes=None, level=1, random_id=1):
    hypernym_map = {}
    imagenet_classes = get_imagenet_classes(hypernym_only=True)
    for synset in imagenet_classes:
        hypernym_map[synset] = find_all_hypernyms(synset, level)

    filtered_classes = find_map_without_duplicates(hypernym_map, random_subset=number_of_classes)
    print("Number of classes:", len(filtered_classes))
    wordnet_id_map = {}
    for key, value in filtered_classes.items():
        key_id = f"{key.pos()}{key.offset():08d}"
        wordnet_id_map[key_id] = [f"{hyper.pos()}{hyper.offset():08d}" for hyper in value]

    file_name = f'../zero_shot_analyze/imagenet/{len(filtered_classes)}_hypernym_level_{level}_{random_id}'

    dict_to_store = {
        "original_id": list(wordnet_id_map.keys()),
        "related_id": list(wordnet_id_map.values())
    }

    with open(f"{file_name}.json", 'w') as f_:
        json.dump(dict_to_store, f_, indent=4)

    build_imagenet_csv(list(wordnet_id_map.keys()), image_to_test=30, csv_name=f"{file_name}.csv")

    return len(filtered_classes), file_name.split("/")[-1]


def imagenet_mixed(number_of_classes=None, level=1, random_id=1):
    mixed_map = {}
    imagenet_classes = get_imagenet_classes(have_both=True)

    for synset in imagenet_classes:
        # Get the synset name (as the original ImageNet class label)
        original_name = synset.name().split(".")[0].replace("_", " ").replace("-", " ")

        # Get synonyms for this synset
        synonyms = find_all_synonyms(synset, original_name)
        # Get hypernyms for this synset
        hypernyms = find_all_hypernyms(synset, level)

        # Combine synonyms and hypernyms (store them in a set to remove duplicates)
        related_terms = set(synonyms)
        related_terms.update([f"{hyper.pos()}{hyper.offset():08d}" for hyper in hypernyms])

        # Add the related terms to the mixed_map with the synset as the key
        if related_terms:
            mixed_map[synset] = list(related_terms)  # Store the related terms as a list

        # Use find_map_without_duplicates to filter out any duplicates across the map
    filtered_classes = find_map_without_duplicates(mixed_map, random_subset=number_of_classes)
    print("Number of classes:", len(filtered_classes))

    # Prepare the data to store in JSON
    wordnet_id_map = {}
    for synset, related_terms in filtered_classes.items():
        key_id = f"{synset.pos()}{synset.offset():08d}"
        wordnet_id_map[key_id] = []
        for term in related_terms:
            try:
                wordnet_id_map[key_id].append(f"{term.pos()}{term.offset():08d}")
            except:
                # Treat the term as a synonym (string), skip this part for strings
                wordnet_id_map[key_id].append(term)

    # Prepare the output file name
    file_name = f'../zero_shot_analyze/imagenet/{len(filtered_classes)}_mixed_{random_id}'

    # Prepare the data to store in JSON
    dict_to_store = {
        "original_id": list(wordnet_id_map.keys()),
        "related_id": list(wordnet_id_map.values())
    }

    # Write to JSON file
    with open(f"{file_name}.json", 'w') as f_:
        json.dump(dict_to_store, f_, indent=4)

    # Build CSV with the wordnet ids for testing
    build_imagenet_csv(list(wordnet_id_map.keys()), image_to_test=30, csv_name=f"{file_name}.csv")

    return len(filtered_classes), file_name.split("/")[-1]


def build_other_csv(wordnet_id_list, original_df, id_to_label_map, image_to_test=30, csv_name="result.csv"):
    image_list = []

    def get_paths_by_wordnet_label(df_, label_value, image_to_test=30):
        # Filter rows where 'wordnet_label' equals label_value
        filtered_df = df_[df_['wordnet_label'] == label_value]
        # Extract the 'path' column values and convert to a list
        paths_ = filtered_df['path'].tolist()
        random_files = random.sample(paths_, image_to_test)
        return random_files

    for i in wordnet_id_list:
        paths = get_paths_by_wordnet_label(original_df, i, image_to_test)
        for path in paths:
            image_list.append({"label": id_to_label_map[i], "filepath": path})

    df = pd.DataFrame(image_list)
    print("Number of images:", len(df))
    df.to_csv(csv_name, index=False)


def openimage_csv_synonym(number_of_classes=300, random_id=1):
    # Read the CSV file
    dataframe_all = pd.read_csv("../dataset_management/open_image_noun_filtered.csv")
    all_wordnet_labels = dataframe_all['wordnet_label'].unique()
    print("Number of unique WordNet labels:", len(all_wordnet_labels))

    # Build synset mapping
    synset_map = {}
    for label in all_wordnet_labels:
        pos = label[0]
        offset = int(label[1:])
        synset_map[label] = wn.synset_from_pos_and_offset(pos, offset)

    # Build synonym map (synset -> lemma names)
    synonym_map = {}
    name_to_wordnet_id = {}
    for label, synset in synset_map.items():
        original_name = synset.name().split(".")[0].replace("_", " ").replace("-", " ")
        synonyms = find_all_synonyms(synset, original_name)
        if not synonyms:
            continue
        synonym_map[original_name] = synonyms
        name_to_wordnet_id[original_name] = f"{synset.pos()}{synset.offset():08d}"

    # Filter classes without overlapping synonyms
    filtered_classes = find_map_without_duplicates(
        synonym_map,
        random_subset=number_of_classes,
    )
    print("Number of classes:", len(filtered_classes))
    # Convert to WordNet ID format
    wordnet_id_list = []
    id_to_label_map = {}
    for idx, name in enumerate(filtered_classes.keys()):
        wordnet_id_list.append(name_to_wordnet_id[name])
        id_to_label_map[name_to_wordnet_id[name]] = idx

    file_name = f'../zero_shot_analyze/openimage/{len(filtered_classes)}_synonym_{random_id}'

    dict_to_store = {
        "original_id": wordnet_id_list,
        "related_id": list(filtered_classes.values())
    }

    with open(f"{file_name}.json", 'w') as f_:
        json.dump(dict_to_store, f_, indent=4)

    build_other_csv(wordnet_id_list, dataframe_all, id_to_label_map, image_to_test=30, csv_name=f"{file_name}.csv")
    return len(filtered_classes), file_name.split("/")[-1]


def openimage_csv_hypernym(number_of_classes=None, level=1, random_id=1):
    # Read the CSV file
    dataframe_all = pd.read_csv("../dataset_management/open_image_noun_filtered.csv")
    all_wordnet_labels = dataframe_all['wordnet_label'].unique()
    print("Number of unique WordNet labels:", len(all_wordnet_labels))

    # Build a mapping from label to corresponding synset.
    # Assumes the label format is like 'n12345678' where the first character is POS and the rest is the offset.
    synset_map = {}
    for label in all_wordnet_labels:
        pos = label[0]
        offset = int(label[1:])
        synset_map[label] = wn.synset_from_pos_and_offset(pos, offset)

    # Build the hypernym_map: key is the synset, value is a list of hypernyms up to the specified level.
    hypernym_map = {}
    for label, synset in synset_map.items():
        hypernym_map[synset] = find_all_hypernyms(synset, level)

    cleaned_hypernym_map = find_map_without_duplicates(hypernym_map, random_subset=number_of_classes)
    # transform this to a map with wordnet IDs
    wordnet_id_map = {}
    id_to_label_map = {}
    label = 0
    for key, value in cleaned_hypernym_map.items():
        key_id = f"{key.pos()}{key.offset():08d}"
        wordnet_id_map[key_id] = [f"{hyper.pos()}{hyper.offset():08d}" for hyper in value]
        id_to_label_map[key_id] = label
        label += 1
    print("Number Of Classes", len(cleaned_hypernym_map))
    # all_synset_list = list(cleaned_hypernym_map.keys()) + [hyper for hyper_list in cleaned_hypernym_map.values() for
    #                                                        hyper in hyper_list]
    # print(len(all_synset_list), len(set(all_synset_list)))
    # # print the duplicates
    # print("Duplicates:", [item for item, count in Counter(all_synset_list).items() if count > 1])

    file_name = f'../zero_shot_analyze/openimage/{len(cleaned_hypernym_map)}_hypernym_level_{level}_{random_id}'

    dict_to_store = {
        "original_id": list(wordnet_id_map.keys()),
        "related_id": list(wordnet_id_map.values())
    }

    with open(f"{file_name}.json", 'w') as f_:
        json.dump(dict_to_store, f_, indent=4)

    build_other_csv(list(wordnet_id_map.keys()), dataframe_all, id_to_label_map, image_to_test=30, csv_name=f"{file_name}.csv")
    return len(cleaned_hypernym_map), file_name.split("/")[-1]


if __name__ == "__main__":
    imagenet_csv_synonym(number_of_classes=100)
    # imagenet_mixed(number_of_classes=100, level=1)
    # imagenet_csv_hypernym(number_of_classes=150, level=1)
    # openimage_csv_synonym(number_of_classes=150)
    # openimage_csv_hypernym(level=1)
    # fer_csv_synonym(image_to_test=100)


