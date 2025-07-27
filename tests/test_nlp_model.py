import pandas as pd
from ml.nlp_model import train_nlp_model

def test_train_nlp_model():
    df = pd.DataFrame({
        'review': [
            'Great product!',
            'Horrible service.',
            'Average experience.'
        ],
        'sentiment': ['positive', 'negative', 'neutral']
    })
    results, best_model_name, model, best_cm = train_nlp_model(df, 'sentiment', 'review')
    assert not results.empty
    assert best_model_name == 'DistilBERT'
