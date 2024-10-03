import os
import subprocess
import hashlib
from PIL import Image
from term_image.image import AutoImage
import io
import sys

def set_tokenizer_parallelism(enable_parallelism):
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if enable_parallelism else "false"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def display_image_in_terminal(image):
    """Display a full image in the terminal."""
    term_image = AutoImage(image, width=80)  # Adjust width as needed
    print(term_image)

def open_image(filename):
    """Open the image file with the default image viewer."""
    if os.name == 'posix':  # macOS and Linux
        subprocess.call(('open' if sys.platform == 'darwin' else 'xdg-open', filename))
    elif os.name == 'nt':  # Windows
        os.startfile(filename)

def generate_sha256(image):
    """Generate SHA256 hash for the image."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return hashlib.sha256(img_byte_arr.getvalue()).hexdigest()
