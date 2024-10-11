import joblib
import cv2
import numpy as np
THRESHOLD = 127

def classify_characters(segmented_characters: list, model_filename: str) -> list:
    model = read_model(model_filename)
    segmented_characters = adapt_characters(segmented_characters)
    preds = model.predict(segmented_characters)
    plate = ''.join(preds)
    return plate

def read_model(model_filename: str):
    model = joblib.load(model_filename)
    return model

def adapt_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image, THRESHOLD, 255, cv2.THRESH_BINARY)
    if image_binary.shape != (22,36):
        image_binary = cv2.resize(image_binary, (22,36))
    return image_binary/255

def adapt_characters(segmented_characters):
    for i in range(len(segmented_characters)):
        segmented_characters[i] = adapt_image(segmented_characters[i])
    segmented_characters = np.array(segmented_characters)
    segmented_characters = segmented_characters.reshape(segmented_characters.shape[0], -1)
    return segmented_characters