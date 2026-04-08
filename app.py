import os
import cv2
import hashlib
import shutil
import tempfile
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
from datetime import datetime
import warnings

# Suppress Starlette and Gradio warnings
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio")

# Configure GPU memory growth dynamically
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Error configuring GPU: {e}")

# Configure Matplotlib backend for headless environment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup samples directory
os.makedirs("samples", exist_ok=True)
