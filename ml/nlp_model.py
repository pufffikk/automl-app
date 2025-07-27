import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from ml.ml_functions import confusion_matrix_to_array


def train_nlp_model(df, target_column, text_column):
    df = df.dropna(subset=[text_column, target_column])
    label_enc = LabelEncoder()
    df['label'] = label_enc.fit_transform(df[target_column])

    dataset = Dataset.from_pandas(df[[text_column, 'label']])
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def tokenize(batch):
        return tokenizer(batch[text_column], padding=True, truncation=True)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    train_test = dataset.train_test_split(test_size=0.2)
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(label_enc.classes_)
    )

    args = TrainingArguments(
        output_dir='./results',
        per_device_train_batch_size=8,
        num_train_epochs=2,
        logging_steps=10,
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_test['train'],
        eval_dataset=train_test['test']
    )

    trainer.train()

    preds = trainer.predict(train_test['test'])
    pred_logits = preds.predictions

    if len(pred_logits.shape) == 1:
        pred_labels = np.array([np.argmax(pred_logits)])
    else:
        pred_labels = np.argmax(pred_logits, axis=1)

    true_labels = np.array(train_test['test']['label'])
    accuracy = (pred_labels == true_labels).mean()
    cm = confusion_matrix(true_labels, pred_labels, labels=range(len(label_enc.classes_)))
    cm_flattered = confusion_matrix_to_array(cm)
    return (pd.DataFrame([{'Model': 'DistilBERT', 'Accuracy': accuracy, 'ROC AUC': None, 'Confusion Matrix': cm_flattered}]),
            'DistilBERT', model, cm)
