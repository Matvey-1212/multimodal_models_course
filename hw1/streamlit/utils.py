import streamlit as st
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel, SiglipModel, AutoImageProcessor
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPImageProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration

def get_nearest_img(q_vec, 
                    index, 
                    q_idx=-1,
                    ds_text=None, 
                    index_text=None, 
                    img_to_vis=10, 
                    no_repeate=True, 
    ):
        
    D, I = index.search(q_vec, img_to_vis + 1)
    I = I[0].tolist()

    if no_repeate and q_idx in I:
        I.remove(q_idx)
    I = I[:img_to_vis]
    return I


class ImgTextEmbeddingModel:
    def __init__(self, 
                 model_name,
                 device=None,
                 token_max_len=77,
                 half=True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.token_max_len = token_max_len
        self.processor = AutoProcessor.from_pretrained(model_name,use_fast=True)
        if "clip" in model_name:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.processor = CLIPImageProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)
            self.model_type = "clip"
        elif "siglip" in model_name:
            self.model = SiglipModel.from_pretrained(model_name)
            self.model_type = "siglip"
        else:
            self.model = AutoModel.from_pretrained(model_name)
            self.model_type = "vision"
        self.model = self.model.to(self.device).eval()

        if half and self.device=='cuda':
            self.model = self.model.half()

    def __call__(self, data, infer_type='img'):
        with torch.no_grad():
            if infer_type=='img':
                inputs = self.processor(images=data, return_tensors="pt").to(self.device)
                if self.model_type in["clip", "siglip"]:
                    feats = self.model.get_image_features(**inputs)
                else: 
                    feats = self.model(**inputs).last_hidden_state[:, 0]
            else:
                if self.model_type == "siglip":
                    inputs = self.processor(
                        text=data, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=self.token_max_len
                    ).to(self.device)
                elif self.model_type == "clip":
                    inputs = self.tokenizer(data, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    
                feats = self.model.get_text_features(**inputs)
                
            return torch.nn.functional.normalize(feats, dim=-1)
        
    def process_dataset(self, ds, batch_size=10, infer_type='img'):
        progress_text = 'Обработка датасета...'
        progress_bar = st.progress(0, text=progress_text)
        embeddings = []

        for i in range(0, len(ds), batch_size):
            if infer_type == 'img':
                batch = [ds[j]["image"] for j in range(i, min(i + batch_size, len(ds)))]
            else:
                batch = [ds[j]["caption"] for j in range(i, min(i + batch_size, len(ds)))]

            feats = self(batch, infer_type).cpu().numpy()
            embeddings.append(feats)

            progress = int((i + batch_size) / len(ds) * 100)
            progress_bar.progress(min(progress, 100), text=f"{progress_text} ({progress}%)")

        embeddings = np.vstack(embeddings).astype("float32")
        return embeddings

class ImgCaptionModel:
    def __init__(self, 
                 model_name,
                 device=None,
                 half=True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if 'blip2' in model_name:
            self.is_blip2 = True
            self.processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_name)
        else:
            self.is_blip2 = False
            self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)

        self.model = self.model.to(self.device).eval()
        if half:
            self.model = self.model.half()

    def __call__(self, images, promt=None, max_length=30):
        with torch.no_grad():
            inputs = self.processor(
                images=images, 
                text=[promt] * len(images) if promt is not None else None, 
                return_tensors="pt"
            ).to(self.device)

            if self.is_blip2:
                outputs = self.model.generate(**inputs, max_new_tokens=max_length)
                captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
            else:
                outputs = self.model.generate(**inputs, max_length=max_length)
                captions = [self.processor.decode(o, skip_special_tokens=True) for o in outputs]

        return [c.strip() for c in captions]

    def process_dataset(self, ds, promt=None, batch_size=8, max_length=30):
        captions = []
        for i in tqdm(range(0, len(ds), batch_size)):
            batch = [ds[j]["image"] for j in range(i, min(i+batch_size, len(ds)))]
            caps = self(batch, max_length=max_length, promt=promt)
            captions.extend(caps)
        return captions

def omni_search_filter(
    q_emb,
    img_index,
    tags,
    tags_data,
    prefilter_k = 1000,
    top_k = 10,
):
    D, I = img_index.search(q_emb.astype("float32"), prefilter_k)
    D, I = D[0], I[0]

    img_embs = np.stack([img_index.reconstruct(int(i)) for i in I])

    mask = np.ones(len(I), dtype=bool)

    for tag_type, tag_values in tags.items():
        if len(tag_values) == 0:
            continue

        tag_info = tags_data[tag_type]
        tag_embs = tag_info.embs
        tag_names = tag_info.value

        target_tag_idxs = [tag_names.index(t) for t in tag_values if t in tag_names]

        tag_scores = img_embs @ tag_embs.T
        best_idx = np.argmax(tag_scores, axis=1)

        tag_mask = np.array([bt in target_tag_idxs for bt in best_idx])
        mask &= tag_mask

    I_filtered = I[mask]
    D_filtered = D[mask]
    
    top_idx = I_filtered[np.argsort(D_filtered)[::-1][:top_k]]

    return top_idx

def omni_search_weighted(
    q_emb,
    img_index,
    tags,
    tags_data,
    tags_w,
    prefilter_k=1000,
    top_k=10,
    w_txt=0.7,
):
    D, I = img_index.search(q_emb.astype("float32"), prefilter_k)
    D, I = D[0], I[0]

    img_embs = np.stack([img_index.reconstruct(int(i)) for i in I])
    img_embs /= np.linalg.norm(img_embs, axis=1, keepdims=True)

    sim_total = w_txt * D.copy()

    for tag_type, tag_values in tags.items():
        if len(tag_values) == 0:
            continue

        tag_info = tags_data[tag_type]
        tag_embs = tag_info.embs
        tag_names = tag_info.value

        
        target_idxs = [tag_names.index(t) for t in tag_values if t in tag_names]
        tag_scores = img_embs @ tag_embs.T

        tag_sim = np.max(tag_scores[:, target_idxs], axis=1)
        sim_total += tags_w[tag_type] * tag_sim

    top_idx = I[np.argsort(sim_total)[::-1][:top_k]]
    return top_idx

def omni_search_promt(
    model,
    query_text,
    img_index,
    tags,
    prefilter_k=1000,
    top_k=10,
):
    parts = [query_text]

    for tag_type, tag_values in tags.items():
        if len(tag_values) == 0:
            continue
        joined = ", ".join(tag_values)
        parts.append(f"{tag_type}: {joined}")

    q_aug = ", ".join(parts)

    q_emb = model(q_aug, infer_type='text').cpu().numpy()
    q_emb /= np.linalg.norm(q_emb)

    D, I = img_index.search(q_emb.astype("float32"), prefilter_k)
    I, D = I[0], D[0]
    top_idx = I[np.argsort(D)[::-1][:top_k]]

    return top_idx

