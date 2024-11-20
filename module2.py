import clip
import torch
import pickle
import os
import io
import pandas as pd
import streamlit as st
from thefuzz import process
from PIL import Image
from thefuzz import fuzz
from nltk.util import ngrams
import string

device = "cuda" if torch.cuda.is_available() else "cpu"


#@st.cache_data
def get_model():
    return clip.load("ViT-L/14", device=device)


def generate_emb(path, embeddings_file):
    model, preprocess = get_model()
    images = [os.path.join(path, i) for i in os.listdir(path)]

    # Embedding of the input image
    emb = dict()
    with torch.no_grad():
        for image_name in images:
            image = preprocess(Image.open(image_name)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            image_features /= torch.linalg.norm(image_features,
                                                dim=-1, keepdims=True)
            emb[os.path.split(image_name)[1]] = image_features

    with open(embeddings_file, "wb") as f:
        pickle.dump(emb, f)


#@st.cache_data
def get_embedding(embeddings_file):
    # generate_emb(cropped)
    with open(embeddings_file, 'rb') as f:
        return pickle.load(f)


def search_img(target, emb, img2img_search, output_file, num_results):
    model, preprocess = get_model()
    cos = torch.nn.CosineSimilarity(dim=0)
    with torch.no_grad():
        result = {}
        if img2img_search:
            image = preprocess(Image.open(io.BytesIO(target))).unsqueeze(0).to(device)
            features = model.encode_image(image)
            features /= torch.linalg.norm(features, dim=-1, keepdims=True)
        else:
            text = clip.tokenize(target).to(device)
            features = model.encode_text(text)

        for i, img in emb.items():
            sim = cos(img[0], features[0]).item()
            sim = (sim + 1) / 2
            result[i] = sim

        sorted_value = sorted(result.items(), key=lambda x: x[1],
                              reverse=True)  # sort embeddings by similarity score
        sorted_res = dict(sorted_value)
        top = list(sorted_res.keys())[:num_results]
        images_found = top

        with open(output_file, 'w') as f:  # write filenames and similarity scores to text file
            for im in images_found:
                f.write(im + '\n')
    return images_found


def find_caption(target, captions_file, output_file, num_results):
    placenames = [target]
    captions_df = pd.read_csv(captions_file)
    columns = captions_df.loc[:,
              captions_df.columns != 'Page'].columns.tolist()
    columns = [item for item in columns if item.startswith('text')]
    d = dict()
    for placename in placenames:
        for i, row in captions_df.iterrows():
            for c in columns:
                tuples = process.extract(placename, [row[c]])
                if len(tuples) != 0:
                    d[row['Page']] = process.extract(placename, [row[c]])[0]
    sorted_value = sorted(d.keys(), key=lambda x: d[x][1], reverse=True)
    found_pages = sorted_value[:num_results]
    with open(output_file, 'w') as f:
        for im in found_pages:
            f.write(im + '\n')
    return found_pages  ### this list should be empty if nothing is found


def search_captions(query, captions_file):
    all_captions = pd.read_csv(captions_file)
    columns = all_captions.loc[:,
              all_captions.columns != 'Page'].columns.tolist()
    columns = [item for item in columns if item.startswith('text')]

    n = len(query.split())

    captions = list()
    all_scores = list()
    j = 0
    for r, row in all_captions.iterrows():
        for c in columns:
            if isinstance(row[c], str):
                caption = row[c]
                captions.append(caption.lower())
                caption = caption.translate(str.maketrans('', '', string.punctuation))
                n_grams = list()
                for i in range(n):
                    n_grams.extend(list(ngrams(caption.split(), i + 1)))
                n_grams = [' '.join(tup) for tup in n_grams]
                for n_gram in n_grams:
                    all_scores.append({'image': str(r), 'sentence': str(j), 'ngram': n_gram,
                                       'score': (fuzz.token_sort_ratio(n_gram, query))})
                j += 1

    new_scores = sorted(all_scores, key=lambda d: d['score'], reverse=True)
    new_scores = sorted(new_scores, key=lambda d: d['image'])
    all_scores = [new_scores[0]]
    for item in new_scores:
        if item['image'] != all_scores[len(all_scores) - 1]['image']:
            all_scores.append(item)
    all_scores = sorted(all_scores, key=lambda d: d['score'], reverse=True)
    sorted_images = list()
    for score in all_scores:
        sorted_images.append(int(score['image']))
    filtered_df = all_captions.reindex(sorted_images)
    image_names = filtered_df['Page'].tolist()
    return image_names
