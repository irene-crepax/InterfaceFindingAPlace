import os
import json
import streamlit as st
from pathlib import Path
import sys
from streamlit_image_select import image_select

sys.path.append(str(Path(__file__).resolve().parent.parent))
from module1 import Preprocessing
from module2 import get_embedding, search_img, find_caption, search_captions, generate_emb

st.set_page_config(layout="wide")
st.header('Search Demo')
cols = st.columns(2)

embeddings_file = 'illustration_archive_embeddings.pkl'
images_path = "illustration_archive"
cropped = 'illustration_archive_cropped'


#st.markdown("# Search Demo")
st.sidebar.header("Search Demo")


def init(images_path, embeddings_file, cropped):
    if 'demo_page_result' not in st.session_state:
        st.session_state['demo_page_result'] = None
    if 'demo_image_result' not in st.session_state:
        st.session_state['demo_image_result'] = None

    p = Preprocessing(images_path)
    #generate_emb(cropped, embeddings_file)
    emb = get_embedding(embeddings_file)

    return emb


def setup(emb):
    with cols[0]:
        st.write("Search Related Images")
        #
        # num_results = st.number_input(
        #     "Number of results", value=30, placeholder="Type a number..."
        # )
        #
        keyword_input = st.text_input('Search images by keywords in the captions')
        keyword_search_button = st.button('Search', key="search_by_caption")
        if keyword_input and len(keyword_input) > 0 and keyword_search_button:
            st.session_state.demo_image_result = None
            st.session_state.demo_page_result = search_captions(query=keyword_input, captions_file='illustration_archive_captions.csv')
        image_file = st.file_uploader("Search for similar images", type=["jpg", "jpeg", "png"])
        image_search_button = st.button('Search', key="search_by_image")
        if image_file and image_search_button:
            st.session_state.demo_page_result = None
            st.session_state.demo_image_result = search_img(target=image_file.read(), emb=emb,
                                                       img2img_search=True, output_file='image.txt', num_results=None)
        #
        keyword_input = st.text_input(
            'Search the content of images using keywords or phrases(e.g. "cat" or "a cat playing with a ball")')
        keyword_search_button = st.button('Search', key="search_by_content")
        if keyword_input and len(keyword_input) > 0 and keyword_search_button:
            st.session_state.demo_page_result = None
            st.session_state.demo_image_result = search_img(target=keyword_input, emb=emb,
                                                       img2img_search=False, output_file='image.txt', num_results=None)
    with cols[1]:
        if st.session_state.demo_page_result is not None:
            if len(st.session_state.demo_page_result) > 0:
                images_list = os.listdir('illustration_archive')
                for p in st.session_state.demo_page_result:
                    subset = [os.path.join('illustration_archive', i) for i in images_list if p in i]
                    img = image_select(label=f"Found in {p}", images=subset, key=p)
                    if img:
                        st.write(img)
                        st.image(img)
            else:
                st.warning('Not found')

        if st.session_state.demo_image_result is not None:

            if len(st.session_state.demo_image_result) > 0:
                img = image_select(
                    label="Select a image",
                    images=[os.path.join('illustration_archive_cropped', i) for i in st.session_state.demo_image_result],

                )
                if img:
                    st.write(img)
                    st.image(img)
                    #with open(os.path.join('illustration_archive_labels/',
                    #                       '_'.join(os.path.basename(img).split('.')[0].split('_')[1:]) + '.json'),
                    #          'r') as f:
                    #    st.write(json.load(f))
            else:
                st.warning('Not found')


try:
    emb = init(images_path, embeddings_file, cropped)
except Exception as e:
    emb = None
    st.error(e)

if emb:
    setup(emb)
else:
    st.error("No embedding found!")
