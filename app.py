import base64
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager

from utils import transform_annotations, inference, transform_scale, get_heatmap


def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}

    color_annotation_app()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp made by <a href="https://twitter.com/andfanilo">Manfred Michael</a></h6>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="margin: 0.75em 0;"><a href="https://www.buymeacoffee.com/andfanilo" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a></div>',
            unsafe_allow_html=True,
        )

def full_app():
    st.sidebar.header("Configuration")
    st.markdown(
        """
    Draw on the canvas, get the drawings back to Streamlit!
    * Configure canvas in the sidebar
    * In transform mode, double-click an object to remove it
    * In polygon mode, left-click to add a point, right-click to close the polygon, double-click to remove the latest point
    """
    )

def color_annotation_app():
    st.markdown(
        """
    #
    """
    )
    
    try:
        bg_image = Image.open("img/annotation.jpeg")
        im = ImageManager('img/annotation.jpeg')
    except:
        im = None
        bg_image = None 

    # bg_image = None 
    image_file = st.file_uploader('Upload your image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    if image_file is not None: 
        bg_image = Image.open(image_file).convert('RGB')
        bg_image.save('img/annotation.jpeg')

    im = ImageManager('img/annotation.jpeg')
    resized_img = im.resizing_img()
    resized_rects = im.get_resized_rects()


    if im:
        rects = st_img_label(resized_img, box_color="red", rects=resized_rects)
        if len(rects) > 0:
            df = pd.DataFrame(rects).drop('label', axis=1)
            st.dataframe(df)
            n_objects = len(rects)
            if  n_objects < 3:
                st.write('annotate at least {} more object(s)'.format(3 - n_objects))
                pass
            else:
                with st.form("my_form"):

                    # Every form must have a submit button.
                    count_button_clicked = st.form_submit_button("Count objects")
                    heatmap_button_clicked = st.form_submit_button("Show heatmaps")
                    if count_button_clicked:
                        annotations = transform_annotations(df)
                        prediction = inference(annotations)
                        st.write(prediction)
                    elif heatmap_button_clicked:
                        annotations = transform_annotations(df)
                        prediction, heatmap = get_heatmap(annotations)
                        st.write(f"predicted count: {round(prediction)}")
                        st.image(heatmap)


if __name__ == "__main__":
    idm = ImageDirManager('img')


    st.set_page_config(
        page_title="Class-agnostic Counting Model", page_icon=":pencil2:"
    )
    st.title("Class-agnostic Counting Demo")
    st.sidebar.subheader("Configuration")
    main()
