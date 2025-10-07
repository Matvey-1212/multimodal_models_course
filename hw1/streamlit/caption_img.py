import pandas as pd
import os
import streamlit as st
import faiss
import random
from PIL import Image

from utils import ImgTextEmbeddingModel, get_nearest_img, ImgCaptionModel

def toggle_a():
    if st.session_state.cb_a:
        st.session_state.cb_b = False

def toggle_b():
    if st.session_state.cb_b:
        st.session_state.cb_a = False

class CaptionSearch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.index_path = os.path.join(cfg.CUR_DIR, cfg.INDEX_FOLLDER)
        self.txt_path = os.path.join(cfg.CUR_DIR, cfg.INDEX_FOLLDER)

    def get_tab_content(self, tab):
        tab.title("Find image by caption")

        controls = tab.columns(3)
        model_name = controls[2].selectbox("Emb Model:", self.cfg.OMNI_MODELS, key='model_selector_2')
        selected_type = controls[2].multiselect("search type:", ['caption->img', 'caption->caption'], key='type_selector_2')

        if model_name and selected_type:


            if model_name not in st.session_state.models:
                st.session_state.models[model_name] = ImgTextEmbeddingModel(model_name, self.cfg.DEVICE)


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

            save_path = f"{self.cfg.CUR_DIR}/{self.cfg.INDEX_FOLLDER}/{model_name.replace('/','_')}_{len(self.cfg.DATASET)}_text.index"
            if save_path not in st.session_state.indexes:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                if not os.path.isfile(save_path):
                    model = st.session_state.models[model_name]
                    embeddings = model.process_dataset(self.cfg.CAPTION_DATASET, batch_size=100, infer_type='text')
                    index = faiss.IndexFlatIP(embeddings.shape[1])
                    index.add(embeddings)
                    faiss.write_index(index, save_path)
                else:
                    index = faiss.read_index(save_path)

                st.session_state.indexes[save_path] = index

            text_input = controls[0].text_area(
                "Введите описание картинки",
                placeholder="Например: 'a painting depicting a woman in a white dress'",
                key="text_query_2",
                height=150
            )
            random_button = controls[0].button('Случайный запрос')
            img_N = controls[2].slider("top-k:", 1, 100, 10, key='top-k-12r')


            if random_button:
                idx = random.randint(0, len(self.cfg.CAPTION_DATASET) - 1)
                text_input = self.cfg.CAPTION_DATASET[idx]['caption']

            if text_input:
                tab.markdown("### Входной запрос")
                tab.write(f'{text_input}')
                tab.markdown("---")

                cont = tab.columns(2)
                
                for k, search_type in enumerate(selected_type):
                    index_path = f"{self.cfg.CUR_DIR}/{self.cfg.INDEX_FOLLDER}/{model_name.replace('/','_')}_{len(self.cfg.DATASET)}.index"
                    img_index = st.session_state.indexes[index_path]
                    
                    index_path = f"{self.cfg.CUR_DIR}/{self.cfg.INDEX_FOLLDER}/{model_name.replace('/','_')}_{len(self.cfg.DATASET)}_text.index"
                    text_index = st.session_state.indexes[index_path]

                    model = st.session_state.models[model_name]

                    text_emb = model([text_input], infer_type='text').cpu().numpy()
                    if search_type == 'caption->img':
                        index = img_index
                    elif search_type == 'caption->caption':
                        index = text_index

                    indices = get_nearest_img(text_emb, index, img_to_vis=img_N)

                    results_placeholder = cont[k].container()
                    with results_placeholder.expander(f"Тип поиска: {search_type}", expanded=True):
                        cols = st.columns(min(1, len(selected_type)))
                        for i, idx in enumerate(indices):
                            sample = self.cfg.DATASET[int(idx)]
                            img = sample["image"]

                            loc_caption = self.cfg.CAPTION_DATASET[int(idx)]['caption']

                            with cols[i % len(cols)]:
                                st.image(img, width=300)
                                st.caption(f" {loc_caption}")

        return tab
