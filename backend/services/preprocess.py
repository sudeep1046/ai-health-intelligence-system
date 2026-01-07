import numpy as np
import cv2
from PIL import Image
from io import BytesIO

IMAGE_SIZE = (224, 224)

def preprocess_image_bytes(image_bytes):
    """
    Convert raw image bytes into a normalized tensor
    """

    # Read image from bytes
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Convert to NumPy array
    image = np.array(image)

    # Resize to model input size
    image = cv2.resize(image, IMAGE_SIZE)

    # Normalize pixel values
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image
