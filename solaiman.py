from transformers import RobertaForSequenceClassification, RobertaTokenizer
import json
import torch
from urllib.parse import urlparse, unquote

import pandas as pd
import os
import time

import numpy as np
from sklearn.utils import shuffle

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

model_name = 'roberta-large'
print(f"Defining model {model_name}")

model = RobertaForSequenceClassification.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)
device='cuda' if torch.cuda.is_available() else 'cpu'

device = 'cpu'

def evaluate(query):
    tokens = tokenizer.encode(query)
    all_tokens = len(tokens)
    tokens = tokens[:tokenizer.max_len - 2]
    used_tokens = len(tokens)
    tokens = torch.tensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]).unsqueeze(0)
    mask = torch.ones_like(tokens)
    with torch.no_grad():
        logits = model(tokens.to(device), attention_mask=mask.to(device))[0]
        probs = logits.softmax(dim=-1)

    fake, real = probs.detach().cpu().flatten().numpy().tolist()

    # Original:
#    return json.dumps(dict(
#         all_tokens=all_tokens,
#         used_tokens=used_tokens,
#         real_probability=real,
#         fake_probability=fake
#     ))

# Changed to return only the binary classification result. 1 if sentence is likely machine-generated.
    return (fake > 0.5, fake, real)

def initialize(checkpoint):
#     if checkpoint.startswith('gs://'):
#         print(f'Downloading {checkpoint}', file=sys.stderr)
#         subprocess.check_output(['gsutil', 'cp', checkpoint, '.'])
#         checkpoint = os.path.basename(checkpoint)
#         assert os.path.isfile(checkpoint)

    print(f'Loading checkpoint from {checkpoint}')
    data = torch.load(checkpoint, map_location=device)
    model.load_state_dict(data['model_state_dict'])
    model.eval()

print("... initializing model")
initialize('detector-large.pt')
# initialize('detector-base.pt')

x = evaluate("hello world")
print(f"test on 'hello world': {x}")


# Dataset 
data_path = "./../../get_text_detect_space/datasets/GPT2vsWebText/"
datasets = sorted([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])

print(f"Datasets: {datasets}")

ds_hw_filename = 'webtext.train.jsonl'
ds_hw = pd.read_json(os.path.join(data_path, ds_hw_filename), lines = True)       

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
    y_ds, preds = model.predict(dataset)
    confusion = confusion_matrix(y_ds, preds)
    score = score_func(y_ds, preds)

    # Return the overall scores and other experimental info:
    return {
        'model': model,
#        'labeled_outputs': y_ds,
#        'predictions': preds,
        'confusion_matrix': confusion,
        'score': score }

class Wrapper:
    def __init__(self):
        pass

    def predict(self, dataset, truncateAt=512):
        y_pred = []
        y_dataset = []

        totalsize = len(dataset)
        progressupdate = max(min(100, int(totalsize/10)), 1)
        print(f"progupd: {progressupdate}; m(m(100, {int(totalsize/100)}), 1)")
        for index, row in dataset.iterrows():
            if not row['X']:
                print("Skipping sentence because it appears to be empty")
                continue
            try:
                truncated_sentence = row['X'][:truncateAt] if len(row['X']) > truncateAt else row['X']
                (is_generated, fake, real) = evaluate(truncated_sentence)
                y_pred.append(1 if is_generated else 0)
                y_dataset.append(row['y'])

                if (index) % progressupdate == 0:
                    print(f"... progress: {index} ({index / totalsize})")
            except:
                print(f"Error while processing the query: {row['X']}")
                break
        return (y_dataset, y_pred)

solaiman_detector = Wrapper()
hw_pd = create_dataset(ds_hw["text"], is_machine_generated=False)


print("Starting evaluation...")
solaiman_on_hw_10k = gtc_evaluate(hw_pd[:10000], solaiman_detector, score_func=safe_accuracy)

print(f"Finished. Solaiman on HW10k: {solaiman_on_hw_10k}")

