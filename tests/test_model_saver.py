from utils.model_saver import save_model_as_pickle
from sklearn.linear_model import LogisticRegression

def test_save_model():
    model = LogisticRegression()
    buffer = save_model_as_pickle(model)
    assert buffer is not None
    assert buffer.getbuffer().nbytes > 0
