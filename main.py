import numpy as np
import gradio as gr

from launch import prepare_environment
from modules import simple_ui


def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189], 
        [0.349, 0.686, 0.168], 
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

if __name__ == '__main__':
    # demo = gr.Interface(sepia, gr.Image(shape=(200, 200)), "image")
    
    # prepare_environment()
    demo = simple_ui.create_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=6006)
