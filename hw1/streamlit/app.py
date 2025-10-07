import streamlit as st
from config import Config
from img_img import ImgSearch
from caption_img import CaptionSearch
from omni_search import OMNISearch

cfg = Config()

def main():
    st.set_page_config(layout='wide')
    st.title("IMG Search")

    tabs = st.tabs([
        'IMG->IMG', 'CAPTION->IMG', 'OMNI-SEARCH'
    ])
    ImgSearch_tab = ImgSearch(cfg)
    ImgSearch_tab.get_tab_content(tabs[0])

    CaptionSearch_tab = CaptionSearch(cfg)
    CaptionSearch_tab.get_tab_content(tabs[1])

    OMNISearch_tab = OMNISearch(cfg)
    OMNISearch_tab.get_tab_content(tabs[2])
    
if __name__ == "__main__":
    main()