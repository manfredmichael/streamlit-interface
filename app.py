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
from utils import setup_image_directory, transform_annotations, inference, transform_scale, get_heatmap



def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}

    color_annotation_app()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp forked from <a href="https://twitter.com/andfanilo">@andfanilo</a></h6>',
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
    except:
        bg_image = None 

    # bg_image = None 
    image_file = st.file_uploader('Upload your image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)
    if image_file is not None: 
        bg_image = Image.open(image_file).convert('RGB')
        import os
        st.write(os.listdir())
        bg_image.save('img/annotation.jpeg')

    label_color = (
        st.sidebar.color_picker("Annotation color: ", "#EA1010") + "77"
    )  # for alpha from 00 to FF
    label = st.sidebar.text_input("Label", "Default")
    mode = "transform" if st.sidebar.checkbox("Adjust ROIs", False) else "rect"

    if bg_image:
        canvas_result = st_canvas(
            fill_color=label_color,
            stroke_width=3,
            background_image=bg_image,
            height=320,
            width=512,
            drawing_mode=mode,
            key="color_annotation_app",
        )
        if canvas_result.json_data is not None:
            df = pd.json_normalize(canvas_result.json_data["objects"])
            df = transform_scale(df, bg_image.size)
            n_objects = len(canvas_result.json_data['objects'])
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
                        st.write(f"predicted count: {prediction}")
                        st.image(heatmap)


            if len(df) == 0:
                return
            st.session_state["color_to_label"][label_color] = label
            df["label"] = df["fill"].map(st.session_state["color_to_label"])
            st.dataframe(df[["top", "left", "width", "height", "fill", "label"]])
            # st.dataframe(df)

def count_objects():
    st.write('bla bla')


if __name__ == "__main__":
    # setup_image_directory()
    st.set_page_config(
        page_title="Class-agnostic Counting Model", page_icon=":pencil2:"
    )
    st.title("Drawable Canvas Demo")
    st.sidebar.subheader("Configuration")
    main()
