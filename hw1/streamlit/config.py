import os
from dataclasses import dataclass
from datasets import load_dataset
import torch
import streamlit as st

st.session_state.indexes = {}
st.session_state.models = {}

@dataclass
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CUR_DIR = os.path.dirname(os.getcwd())
    INDEX_FOLLDER = 'indexes'
    TAGS_FOLLDER = 'tags'
    DATASET = load_dataset("huggan/wikiart", split="train")
    CAPTION_DATASET = load_dataset(f"mtvA/wikiart-captions_{len(DATASET)}", split="train")
    IMG_MODELS = [
        'facebook/dinov2-base',
        'google/siglip2-base-patch16-224'
    ]
    OMNI_MODELS = [
        'openai/clip-vit-base-patch16'
    ]