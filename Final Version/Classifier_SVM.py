import pickle


def classify_characters(segmented_characters: list, model_filename: str) -> list:
    model = read_model(model_filename)
    preds = model.predict(segmented_characters)
    plate = ''.join(preds)
    return plate

def read_model(model_filename: str):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

