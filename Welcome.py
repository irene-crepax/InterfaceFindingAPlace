import streamlit as st
import os
import json
import tkinter as tk
from tkinter import filedialog

from streamlit_image_select import image_select
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from module1 import Preprocessing
from module2 import get_embedding, search_img, find_caption, search_captions, generate_emb
import clip
import torch
import pickle
import io
import pandas as pd
from thefuzz import process
from PIL import Image
from thefuzz import fuzz
from nltk.util import ngrams
import string
import imutils
import pytesseract
import cv2



from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

st.write("# Welcome to Finding your Place! 👋")

st.sidebar.header("Welcome")

st.markdown(
    """
    Welcome to Finding your Place, the application that allows you to automate searching on digitised printed material.
"""
)