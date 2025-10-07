import pandas as pd
import os
import streamlit as st
import faiss
import random
from PIL import Image

from utils import ImgTextEmbeddingModel, get_nearest_img

class ImgSearch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.index_path = os.path.join(cfg.CUR_DIR, cfg.INDEX_FOLLDER)

    def get_tab_content(self, tab):
        tab.title("Find image by another image")

        controls = tab.columns(3)
        selected_models = controls[2].multiselect("Emb Models:", self.cfg.IMG_MODELS, key='model_selector_1')

        if selected_models:

            for model_name in selected_models:
                if model_name not in st.session_state.models:
                    st.session_state.models[model_name] = ImgTextEmbeddingModel(model_name, self.cfg.DEVICE)


            for model_name in selected_models:
                save_path = f"{self.cfg.CUR_DIR}/{self.cfg.INDEX_FOLLDER}/{model_name.replace('/','_')}_{len(self.cfg.DATASET)}.index"

                if save_path not in st.session_state.indexes:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if not os.path.isfile(save_path):
                        model = st.session_state.models[model_name]
                        embeddings = model.process_dataset(self.cfg.DATASET, batch_size=100)
                        index = faiss.IndexFlatIP(embeddings.shape[1])
                        index.add(embeddings)
                        faiss.write_index(index, save_path)
                    else:
                        index = faiss.read_index(save_path)

                    st.session_state.indexes[save_path] = index

            uploaded_file = controls[0].file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])
            random_button = controls[0].button('Случайное изображение')
            img_N = controls[2].slider("top-k:", 1, 100, 10, key='top-k-1')

            image = None
            meta = {}

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                meta = {"style": "-", "genre": "-", "artist": "-", 'q_idx':-1}
            elif random_button:
                idx = random.randint(0, len(self.cfg.DATASET) - 1)

                sample = self.cfg.DATASET[idx]
                image = sample["image"]
                meta = {
                    "style": self.cfg.DATASET.features["style"].int2str(sample.get("style", 0)),
                    "artist": self.cfg.DATASET.features["artist"].int2str(sample.get("artist", 0)),
                    "genre": self.cfg.DATASET.features["genre"].int2str(sample.get("genre", 0)),
                    'q_idx': idx
                }

            if image is not None:
                tab.markdown("### Входное изображение")
                tab.image(
                    image, 
                    caption=f"Style: {meta['style']} | Genre: {meta['genre']} | Artist: {meta['artist']}",
                    width=300
                )
                tab.markdown("---")

                cont = tab.columns(2)
                
                for k, model_name in enumerate(selected_models):
                    index_path = f"{self.cfg.CUR_DIR}/{self.cfg.INDEX_FOLLDER}/{model_name.replace('/','_')}_{len(self.cfg.DATASET)}.index"

                    index = st.session_state.indexes[index_path]
                    model = st.session_state.models[model_name]

                    img_emb = model([image], infer_type='img').cpu().numpy()
                    indices = get_nearest_img(img_emb, index, img_to_vis=img_N, q_idx=meta['q_idx'])

                    results_placeholder = cont[k].container()
                    with results_placeholder.expander(f"Модель: {model_name}", expanded=True):
                        cols = st.columns(min(1, len(selected_models)))
                        for i, idx in enumerate(indices):
                            sample = self.cfg.DATASET[int(idx)]
                            img = sample["image"]
                            style = self.cfg.DATASET.features["style"].int2str(sample.get("style", 0))
                            genre = self.cfg.DATASET.features["genre"].int2str(sample.get("genre", 0))
                            artist = self.cfg.DATASET.features["artist"].int2str(sample.get("artist", 0))

                            with cols[i % len(cols)]:
                                st.image(img, width=300)
                                st.caption(f" {style} | {genre} | {artist}")
                

        return tab
