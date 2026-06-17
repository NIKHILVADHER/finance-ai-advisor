import pickle

model = pickle.load(open("model/category_model.pkl", "rb"))

def predict_category(text):
    return model.predict([text])[0]