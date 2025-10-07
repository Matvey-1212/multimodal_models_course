import pandas as pd
import os
import streamlit as st
import faiss
import random
from PIL import Image
from dataclasses import dataclass
from typing import List
import numpy as np

from utils import ImgTextEmbeddingModel, get_nearest_img, omni_search_filter, omni_search_weighted, omni_search_promt

@dataclass
class TagData:
    value: List[str]
    embs: np.array

class OMNISearch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.index_path = os.path.join(cfg.CUR_DIR, cfg.INDEX_FOLLDER)
        self.txt_path = os.path.join(cfg.CUR_DIR, cfg.INDEX_FOLLDER)

    def get_tab_content(self, tab):
        tab.title("Find image by caption")

        controls = tab.columns(3)
        model_name = controls[2].selectbox("Emb Model:", self.cfg.OMNI_MODELS, key='model_selector_3')
        selected_type = controls[2].multiselect("search type:", ['filter_by_tags', 'late_Fusion', 'promt'], key='type_selector_3')

        if model_name and selected_type:
            all_styles = list(map(self.cfg.DATASET.features["style"].int2str, set(self.cfg.DATASET["style"])))
            all_genres = list(map(self.cfg.DATASET.features["genre"].int2str, set(self.cfg.DATASET["genre"])))

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

            save_path = f"{self.cfg.CUR_DIR}/{self.cfg.TAGS_FOLLDER}/{model_name.replace('/','_')}__{len(self.cfg.DATASET)}_tags"
            if save_path not in st.session_state.indexes:
                style_prompts = [
                    f"This painting is in the style of {style.replace('_', ' ')}." for style in all_styles
                ]
                genre_prompts = [
                    f"This is a {genre.replace('_', ' ')}." for genre in all_genres
                ]

                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                if not all([os.path.isfile(save_path+p) for p in ['_style.npy', '_genre.npy']]):
                    model = ImgTextEmbeddingModel(model_name, 'cuda')
                    style_embeddings = model(style_prompts, infer_type='text').cpu().numpy()
                    genre_embeddings = model(genre_prompts, infer_type='text').cpu().numpy()

                    np.save(save_path+'_style.npy', style_embeddings)
                    np.save(save_path+'_genre.npy', genre_embeddings)
                else:
                    style_embeddings = np.load(save_path+'_style.npy')
                    genre_embeddings = np.load(save_path+'_genre.npy')

                style_data = TagData(value=all_styles, embs=style_embeddings)
                genre_data = TagData(value=all_genres, embs=style_embeddings)
                st.session_state.indexes[save_path] = {
                    'genres':genre_data,
                    'styles':style_data
                }

            text_input = controls[0].text_area(
                "Введите описание картинки",
                placeholder="Например: 'a painting depicting a woman in a white dress'",
                key="text_query_3",
                height=150
            )

            random_button = controls[0].button('Случайный входные данные')
            img_N = controls[2].slider("top-k:", 1, 100, 10, key='top-k-13')

            
            controls2 = controls[1].columns(2)
            selected_genres = controls2[0].multiselect(
                "genres:", 
                all_genres, 
                key='genre_selector_3'
            )

            selected_styles = controls2[1].multiselect(
                "styles:", 
                all_styles, 
                key='styles_selector_3'
            )

            if random_button:
                idx = random.randint(0, len(self.cfg.CAPTION_DATASET) - 1)
                text_input = self.cfg.CAPTION_DATASET[idx]['caption']

                gidx = random.randint(0, len(all_genres) - 1)
                sidx = random.randint(0, len(all_styles) - 1)

                selected_genres = [all_genres[gidx]]
                selected_styles = [all_styles[sidx]]

            tags = {
                'genres':selected_genres,
                'styles':selected_styles
            }

            late_Fusion_weughts = {}
            if 'late_Fusion' in selected_type:
                results_placeholder = controls[1].container()
                with results_placeholder.expander(f"Параметры для late_Fusion:", expanded=True):
                    genre_w = st.slider("genre weight:", 0., 1., value=0.2, step=0.3, key='genre_weight_3')
                    style_w = st.slider("style weight:", 0., 1., value=0.2, step=0.3, key='style_weight_3')
                    late_Fusion_weughts = {
                        'genres':genre_w,
                        'styles':style_w
                    }

            if text_input:
                tab.markdown("### Входной запрос")
                tab.write(f'{text_input}')
                tab.write(f'genres: {selected_genres}')
                tab.write(f'styles: {selected_styles}')
                tab.markdown("---")

                cont = tab.columns(3)
                
                for k, search_type in enumerate(selected_type):
                    index_path = f"{self.cfg.CUR_DIR}/{self.cfg.INDEX_FOLLDER}/{model_name.replace('/','_')}_{len(self.cfg.DATASET)}.index"
                    img_index = st.session_state.indexes[index_path]

                    model = st.session_state.models[model_name]

                    tags_path = f"{self.cfg.CUR_DIR}/{self.cfg.TAGS_FOLLDER}/{model_name.replace('/','_')}__{len(self.cfg.DATASET)}_tags"
                    if search_type=='filter_by_tags':
                        text_emb = model([text_input], infer_type='text').cpu().numpy()
                        indices = omni_search_filter(text_emb, img_index, tags, st.session_state.indexes[tags_path])
                    elif search_type=='late_Fusion':
                        text_emb = model([text_input], infer_type='text').cpu().numpy()
                        indices = omni_search_weighted(text_emb, img_index, tags, st.session_state.indexes[tags_path], late_Fusion_weughts)
                    elif search_type=='promt':
                        indices = omni_search_promt(model, text_input, img_index, tags)

                    results_placeholder = cont[k].container()
                    with results_placeholder.expander(f"Тип поиска: {search_type}", expanded=True):
                        cols = st.columns(min(1, len(selected_type)))
                        for i, idx in enumerate(indices):
                            sample = self.cfg.DATASET[int(idx)]
                            img = sample["image"]

                            loc_caption = self.cfg.CAPTION_DATASET[int(idx)]['caption']
                            style = self.cfg.DATASET.features["style"].int2str(sample.get("style", 0))
                            genre = self.cfg.DATASET.features["genre"].int2str(sample.get("genre", 0))


                            with cols[i % len(cols)]:
                                st.image(img, width=300)
                                st.caption(f"{loc_caption} | {style} | {genre}")

        return tab
