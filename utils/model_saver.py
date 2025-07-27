import pickle
import io

def save_model_as_pickle(model):
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    return buffer
