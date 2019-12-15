import math
import numpy as np
import string
import random
import json
import argparse
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from transformers import *
import sys
sys.path.append('../')
from DataLoader.bucket_and_batch import bucket_and_batch
from model.BERT_NL import Encoder as BERT_NL_Encoder
from model.BERT_BiLSTM import Encoder as BERT_BiLSTM_Encoder
from model.BERT_BiLSTM_attn import Encoder as BERT_BiLSTM_attn_Encoder
from model.BERT_attn_BiLSTM import Encoder as BERT_attn_BiLSTM_Encoder
from model.BERT_attn_BiLSTM_attn import Encoder as BERT_attn_BiLSTM_attn_Encoder
from model.BERT_capsule_BiLSTM_attn import Encoder as BERT_capsule_BiLSTM_attn_Encoder
from model.BERT_capsule_BiLSTM_capsule import Encoder as BERT_capsule_BiLSTM_capsule_Encoder
from model.BERT_capsule import Encoder as BERT_capsule_Encoder
import modules.utils as utils
import modules.eval as eval

parser = argparse.ArgumentParser(description='Model Name')
parser.add_argument('--model', type=str, default="BERT_BiLSTM")
flags = parser.parse_args()

model_name = flags.model

print("\n\nTesting Model: {}\n\n".format(model_name))

model_dict = {'BERT_NL': BERT_NL_Encoder,
              'BERT_BiLSTM': BERT_BiLSTM_Encoder,
              'BERT_BiLSTM_attn': BERT_BiLSTM_attn_Encoder,
              'BERT_attn_BiLSTM': BERT_attn_BiLSTM_Encoder,
              'BERT_attn_BiLSTM_attn': BERT_attn_BiLSTM_attn_Encoder,
              'BERT_capsule_BiLSTM_attn': BERT_capsule_BiLSTM_attn_Encoder,
              'BERT_capsule_BiLSTM_capsule': BERT_capsule_BiLSTM_capsule_Encoder,
              'BERT_capsule': BERT_capsule_Encoder}

Encoder = model_dict.get(model_name, BERT_BiLSTM_Encoder)

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

print(device)

if device == T.device('cuda'):
    T.set_default_tensor_type(T.cuda.FloatTensor)
else:
    T.set_default_tensor_type(T.FloatTensor)

random.seed(101)

bnb = bucket_and_batch()

test_batch_size = 32
accu_step = 64/test_batch_size
max_grad_norm = 2

with open('../Processed_Data/test_data.json') as file:

    data = json.load(file)
    # print(len(data["tweets"]))
    test_texts = data["tweets"]
    test_labels = data["labels"]
    test_binary_labels = data["binary_labels"]

with open('../Processed_Data/label_info.json') as file:

    data = json.load(file)

    labels2idx = data["labels2idx"]
    binary_labels2idx = data["binary_labels2idx"]
    label_weights = data["label_weights"]
    binary_label_weights = data["binary_label_weights"]

idx2labels = {v: k for k, v in labels2idx.items()}
binary_idx2labels = {v: k for k, v in binary_labels2idx.items()}


label_weights_idx = [i for i in range(len(labels2idx))]
label_weights = [label_weights[idx2labels[id]] for id in label_weights_idx]
label_weights = T.tensor(label_weights).to(device)

binary_label_weights_idx = [0, 1]
binary_label_weights = [binary_label_weights[binary_idx2labels[id]]
                        for id in binary_label_weights_idx]
binary_label_weights = T.tensor(binary_label_weights).to(device)


model = Encoder(classes_num=len(labels2idx))

model = model.to(device)

parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Parameter Count: ", parameter_count)

parameters = []
BERT_parameters = []
allowed_layers = [11, 10, 9, 8, 7, 6]

for name, param in model.named_parameters():
    if "BERT" not in name:
        parameters.append(param)
        print(name)
        print(param.size())
    else:
        for layer_num in allowed_layers:
            layer_num = str(layer_num)
            if ".{}.".format(layer_num) in name:
                BERT_parameters.append(param)
                print(name)
                print(param.size())
                break


def display(texts, predictions, labels, binary_predictions, binary_labels, label_masks):

    global idx2labels
    global binary_idx2labels

    N = len(texts)
    j = random.choice(np.arange(N).tolist())

    display_text = texts[j]
    display_prediction = idx2labels[predictions[j]]
    display_gold = idx2labels[labels[j]]

    if label_masks[j] == 0:
        display_prediction = display_prediction+" (N\A)"
        display_gold = "(unlabeled)"
    display_binary_prediction = binary_idx2labels[binary_predictions[j]]
    display_binary_gold = binary_idx2labels[binary_labels[j]]

    print("\n\nExample Prediction\n")
    print("Text: {}\n".format(display_text))
    print("Prediction: {}, Gold: {}, Binary Prediction: {}, Binary Gold: {}\n".format(
        display_prediction, display_gold, display_binary_prediction, display_binary_gold))


def predict(text_ids, labels, binary_labels, input_mask, label_mask, train=True):
    global model
    global label_weights
    global binary_label_weights

    with T.no_grad():

        text_ids = T.tensor(text_ids).to(device)
        labels = T.tensor(labels).to(device)
        binary_labels = T.tensor(binary_labels).to(device)
        input_mask = T.tensor(input_mask).float().to(device)
        label_mask = T.tensor(label_mask).float().to(device)

    if train:

        model = model.train()
        binary_logits, logits = model(text_ids, input_mask)
    else:
        model = model.eval()
        binary_logits, logits = model(text_ids, input_mask)

    binary_predictions = np.where(binary_logits.view(-1).detach().cpu().numpy() > 0.5, 1, 0)
    predictions = T.argmax(logits, dim=-1).detach().cpu().numpy()

    loss = utils.cross_entropy(model, logits, labels, binary_logits, binary_labels,
                               label_weights, binary_label_weights, label_mask)

    T.cuda.empty_cache()

    return predictions, binary_predictions, loss


test_batches_texts, test_batches_text_ids, \
    test_batches_labels, test_batches_binary_labels, \
    test_batches_mask, test_batches_label_masks = bnb.bucket_and_batch(
        test_texts, test_labels, test_binary_labels, labels2idx, binary_labels2idx, test_batch_size)

print("Test batches loaded")

display_step = 100
example_display_step = 1 #500
patience = 5

print('Loading pre-trained weights for the model...')

checkpoint = T.load("../Model_Backup/{}.pt".format(model_name))

print(checkpoint["past epoch"])
model.load_state_dict(checkpoint['model_state_dict'])
print('\nRESTORATION COMPLETE\n')


batches_indices = [i for i in range(0, len(test_batches_texts))]
random.shuffle(batches_indices)

total_test_cost = 0
batch_labels = []
batch_binary_labels = []
batch_predictions = []
batch_binary_predictions = []
batch_label_masks = []

for i in range(0, len(test_batches_texts)):

    #if i % display_step == 0:
        #print("Testing Batch {}".format(i+1))

    with T.no_grad():

        predictions, binary_predictions, loss = predict(text_ids=test_batches_text_ids[i],
                                                        labels=test_batches_labels[i],
                                                        binary_labels=test_batches_binary_labels[i],
                                                        input_mask=test_batches_mask[i],
                                                        label_mask=test_batches_label_masks[i],
                                                        train=False)

    cost = loss.item()

    total_test_cost += cost

    predictions = predictions.tolist()
    binary_predictions = binary_predictions.tolist()
    labels = test_batches_labels[i].tolist()
    binary_labels = test_batches_binary_labels[i].tolist()
    label_masks = test_batches_label_masks[i].tolist()

    batch_labels += labels
    batch_binary_labels += binary_labels
    batch_predictions += predictions
    batch_binary_predictions += binary_predictions
    batch_label_masks += label_masks

    if i % example_display_step == 0:

        display(test_batches_texts[i],
                predictions, labels,
                binary_predictions, binary_labels,
                label_masks)

binary_prec, binary_rec, binary_acc = eval.binary_metrics(batch_binary_predictions,
                                                          batch_binary_labels)
prec, rec, acc = eval.multi_metrics(
    batch_predictions, batch_labels, batch_label_masks, idx2labels)
binary_test_F1 = eval.compute_F1(binary_prec, binary_rec)
test_F1 = eval.compute_F1(prec, rec)

test_len = len(test_batches_texts)

avg_test_cost = total_test_cost/test_len

print("\n\nTESTING\n\n")

print("Cost = " +
      "{:.3f}".format(avg_test_cost)+", Binary Precision = " +
      "{:.3f}".format(binary_prec)+", Binary Recall = " +
      "{:.3f}".format(binary_rec)+", Binary F1 = " +
      "{:.3f}".format(binary_test_F1)+", Binary Accuracy = " +
      "{:.3f}".format(binary_acc)+", Multi-Precision = " +
      "{:.3f}".format(prec)+", Multi-Recall = " +
      "{:.3f}".format(rec)+", Multi-F1 = " +
      "{:.3f}".format(test_F1)+", Multi-Accuracy = " +
      "{:.3f}".format(acc))
