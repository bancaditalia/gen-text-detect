import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score

def create_dataset(raw_data, is_machine_generated=True):
    X = np.array(raw_data)
    if is_machine_generated:
        y = np.ones_like(X)
    else:
        y = np.zeros_like(X)
    return pd.DataFrame(data={'X':X, 'y':y})

def concat_and_shuffle(datasets):
    return shuffle(pd.concat(datasets)).reset_index()


def safe_macro_f1(y, y_pred):
    """
    Macro-averaged F1, forcing `sklearn` to report as a multiclass
    problem even when there are just two classes. `y` is the list of
    gold labels and `y_pred` is the list of predicted labels.

    """
    return f1_score(y, y_pred, average='macro', pos_label=None)

def safe_accuracy(y, y_pred):
    return accuracy_score(y, y_pred, normalize=True)

def gtc_evaluate(
        dataset,
        model,
        score_func=safe_accuracy):

    # Predictions if we have labels:
    preds = model.predict(dataset['X'])
    if dataset['y'] is not None:
        y = dataset['y'].tolist()
        confusion = confusion_matrix(y, preds)
        score = score_func(y, preds)
        
    # Return the overall scores and other experimental info:
    return {
        'model': model,
        'predictions': preds, 
        'confusion_matrix': confusion, 
        'score': score }
