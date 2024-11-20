import os
import json
import streamlit as st
import tkinter as tk
from tkinter import filedialog

from streamlit_image_select import image_select
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from module1 import Preprocessing
from module2 import get_embedding, search_img, find_caption, search_captions, generate_emb


def image_select_in_batches(cropped, image_results, batch, p=None):
    if len(image_results) < batch:
        batch = len(image_results)
    return image_select(label="Select an image",
                        images=[os.path.join(cropped, i) for i in image_results[batch - 10:batch]], key=p)


def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_path


st.set_page_config(layout='wide')
st.header('Search your Data')
cols = st.columns(2)
# st.markdown("# Search your Data")
st.sidebar.header("Search your Data")

if 'page_result' not in st.session_state:
    st.session_state['page_result'] = None
if 'image_result' not in st.session_state:
    st.session_state['image_result'] = None
if 'folder_path' not in st.session_state:
    st.session_state['folder_path'] = None
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = None
if 'captions' not in st.session_state:
    st.session_state['captions'] = None
if 'cropped' not in st.session_state:
    st.session_state['cropped'] = None
if 'labels' not in st.session_state:
    st.session_state['labels'] = None
if 'batch' not in st.session_state:
    st.session_state['batch'] = None
with cols[0]:
    st.write('Search your Data')
    folder_select_button = st.button("Select Folder")
    if folder_select_button:
        selected_folder_path = select_folder()
        st.session_state.folder_path = selected_folder_path
        embeddings_file = selected_folder_path.split('/')[-1] + '_embeddings.pkl'
        p = Preprocessing(selected_folder_path)
        if not os.path.isfile(embeddings_file):
            p.create_data()
            generate_emb(p.cropped_path, embeddings_file)
        emb = get_embedding(embeddings_file)
        st.session_state.embeddings = emb
        st.session_state.captions = p.captions_path
        st.session_state.cropped = p.cropped_path
        st.session_state.labels = p.labels_path
    st.write("Search Related Images")
    keyword_input = st.text_input('Search images by keywords in the captions')
    keyword_search_button = st.button('Search', key="search_by_caption")
    if keyword_input and len(keyword_input) > 0 and keyword_search_button:
        st.session_state.image_result = None
        st.session_state.page_result = search_captions(query=keyword_input, captions_file=st.session_state.captions)
        st.session_state.batch = 10
    image_file = st.file_uploader("Search for similar images", type=["jpg", "jpeg", "png"])
    image_search_button = st.button('Search', key="search_by_image")
    if image_file and image_search_button:
        st.session_state.page_result = None
        st.session_state.image_result = search_img(target=image_file.read(), emb=st.session_state.embeddings,
                                                   img2img_search=True, output_file='image.txt', num_results=None)
        st.session_state.batch = 10
    keyword_input = st.text_input(
        'Search the content of images using keywords or phrases(e.g. "cat" or "a cat playing with a ball")')
    keyword_search_button = st.button('Search', key="search_by_content")
    if keyword_input and len(keyword_input) > 0 and keyword_search_button:
        st.session_state.page_result = None
        st.session_state.image_result = search_img(target=keyword_input, emb=st.session_state.embeddings,
                                                   img2img_search=False, output_file='image.txt', num_results=None)
        st.session_state.batch = 10
with cols[1]:
    if st.session_state.page_result is not None:
        if len(st.session_state.page_result) > 0:
            show_more_button = st.button('Show more results', key="show_more_results")
            if show_more_button:
                st.session_state.batch += 10
            img = image_select_in_batches(st.session_state.folder_path, st.session_state.page_result, st.session_state.batch)
            if img:
                st.write(img)
                st.image(img)
        else:
            st.warning('Not found')

    if st.session_state.image_result is not None:
        print(len(st.session_state.image_result))
        if len(st.session_state.image_result) > 0:
            show_more_button = st.button('Show more results', key="show_more_results")
            if show_more_button:
                st.session_state.batch += 10
            img = image_select_in_batches(st.session_state.cropped, st.session_state.image_result, st.session_state.batch)
            if img:
                st.write(img)
                st.image(img)
        else:
            st.warning('Not found')



