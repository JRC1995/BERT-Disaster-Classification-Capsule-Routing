import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import *
import random
import copy

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

if device == T.device('cuda'):
    T.set_default_tensor_type(T.cuda.FloatTensor)
else:
    T.set_default_tensor_type(T.FloatTensor)

# Span Masking: https://arxiv.org/pdf/1907.10529.pdf

text = "Hello world! This is Suraj - the infamous scammer!".lower()

tokenizer = BertTokenizer.from_pretrained('../Classification/Pre_trained_BERT/')

BERT = BertForMaskedLM.from_pretrained('../Classification/Pre_trained_BERT/',
                                       output_hidden_states=False,
                                       output_attentions=False).to(device)

linear1 = nn.Linear(768, 768).to(device)

PAD_id = tokenizer.encode("[PAD]")[0]
MASK_id = tokenizer.encode("[MASK]")[0]

tokens = tokenizer.tokenize(text)

print("tokens: ", tokens)

# replace 15% of text with MASK

span_len = int(.15*len(tokens))
indices = [i for i in range(len(tokens)-span_len)]
random_id = random.choice(indices)

original_tokens = copy.deepcopy(tokens)
gold_span = copy.deepcopy(tokens[random_id:random_id+span_len])

tokens[random_id:random_id+span_len] = ["[MASK]" for i in range(span_len)]

print("Masked tokens: ", tokens)

print("Ground Truth (To Predict): ", gold_span)

text_id = tokenizer.encode(tokens, add_special_tokens=False)
original_text_id = tokenizer.encode(original_tokens, add_special_tokens=False)

print("Text id: ", text_id)

gold_id = tokenizer.encode(gold_span, add_special_tokens=False)

print("Gold id: ", gold_id)

masked_lm_label = []
for c, id in enumerate(text_id):
    if id == MASK_id:
        masked_lm_label.append(original_text_id[c])
    else:
        masked_lm_label.append(-1)

print("Masked LM label: ", masked_lm_label)

batch_text_id = T.tensor([text_id]).long().to(device)
batch_gold_id = T.tensor([gold_id]).long().to(device)
batch_masked_lm_label = T.tensor([masked_lm_label]).long().to(device)

"""
for name, param in BERT.named_parameters():
    print(name)
"""

loss, prediction_scores = BERT(batch_text_id, masked_lm_labels=batch_masked_lm_label)

predictions = T.argmax(prediction_scores, dim=-1)

print("Prediction id: ", predictions[0].cpu().tolist())

prediction = predictions.cpu().numpy().tolist()[0]

print("prediction: ", tokenizer.decode(prediction))

print("loss: ", loss)

# backpropagate from loss
