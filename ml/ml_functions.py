def detect_task_type(df):
    for col in df.columns:
        print(f"COL TYPE {df[col].dtype}")
        if df[col].dtype in ('object', 'string'):
            mean_len = df[col].str.len().mean()
            print(f"mean {mean_len}")
            if mean_len > 30:
                return 'nlp', col
    return 'tabular', None


def confusion_matrix_to_array(cm):
    return [item for sublist in cm for item in sublist]