import pandas as pd
from ml.ml_functions import detect_task_type

def test_detect_tabular():
    df = pd.DataFrame({
        'feature': [1, 2, 3],
        'target': [0, 1, 0]
    })
    task, _ = detect_task_type(df)
    assert task == 'tabular'

def test_detect_nlp():
    df = pd.DataFrame({
        'text': [
            'This is a long review about a product that was amazing.',
            'Another long piece of text describing a negative experience.',
            'Short.'
        ],
        'label': ['pos', 'neg', 'neutral']
    })
    task, text_column = detect_task_type(df)
    assert task == 'nlp'
    assert text_column == 'text'
