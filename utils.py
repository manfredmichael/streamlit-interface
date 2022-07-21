import requests
import json
from PIL import Image
from io import BytesIO, StringIO
from base64 import decodebytes
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

inference_url = 'ubuntu@ec2-108-137-44-0.ap-southeast-3.compute.amazonaws.com'

def add_heatmap_to_image(pil_image, heatmap):
    image = np.array(pil_image).astype(np.float32)
    image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]), interpolation= cv2.INTER_LINEAR)
    image /= 255.0
    image = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
    image = np.clip(image, 0, 1)
    return image

def transform_annotations(df):
    df['x1'] = df['left'] 
    df['x2'] = (df['left'] + df['width'])
    df['y1'] = df['top'] 
    df['y2'] = (df['top'] + df['height']) 

    df['top_left'] = df.apply(lambda x: list([x['x1'], x['y1']]), axis=1) 
    df['bottom_left'] = df.apply(lambda x: list([x['x1'], x['y2']]), axis=1) 
    df['bottom_right'] = df.apply(lambda x: list([x['x2'], x['y2']]), axis=1) 
    df['top_right'] = df.apply(lambda x: list([x['x2'], x['y1']]), axis=1) 
    annotations = df[['top_left', 'bottom_left', 'bottom_right', 'top_right']].values.tolist()
    return annotations 

def transform_scale(df, size):
    if len(df) > 0:
        x_ratio = 512/ size[0]
        y_ratio = 320 / size[1]

        df['left'] = df['left'] / x_ratio
        df['width'] = df['width'] / x_ratio
        df['top'] = df['top'] / y_ratio
        df['height'] = df['height'] / y_ratio

    return df


def inference(annotations, filename):
    result = requests.post(
            f"http://{inference_url}:80/predict",
        files = {'file': open(f"img/{filename}.jpeg", 'rb'),
                 'data': json.dumps({'annotations': annotations})
                },
    )

    return result.json()

def get_heatmap(annotations, filename):
    result = requests.post(
            f"http://{inference_url}:80/heatmap",
        files = {'file': open(f"img/{filename}.jpeg", 'rb'),
                 'data': json.dumps({'annotations': annotations})
                },
    )


    heatmap = result.json()['heatmap']

    heatmap = np.array(heatmap) * 0.5


    colormap = plt.get_cmap('inferno')
    heatmap = (colormap(heatmap)).astype(np.float32)[:,:,:3]


    return result.json()['count'], heatmap 
