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
from transformers import *


import sys
sys.path.append("../")
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

print("\n\nTraining Model: {}\n\n".format(model_name))

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

if device == T.device('cuda'):
    T.set_default_tensor_type(T.cuda.FloatTensor)
else:
    T.set_default_tensor_type(T.FloatTensor)

random.seed(101)

bnb = bucket_and_batch()

val_batch_size = 64
train_batch_size = 64
accu_step = 64/train_batch_size
max_grad_norm = 2

with open('../Processed_Data/train_data.json') as file:

    data = json.load(file)

    train_texts = data["tweets"]  # [0:500]
    train_labels = data["labels"]  # [0:500]
    train_binary_labels = data["binary_labels"]  # [0:500]


with open('../Processed_Data/val_data.json') as file:

    data = json.load(file)

    val_texts = data["tweets"]  # [0:500]
    val_labels = data["labels"]  # [0:500]
    val_binary_labels = data["binary_labels"]  # [0:500]


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

optimizer = T.optim.AdamW([{'params': parameters},
                           {'params': BERT_parameters, 'lr': 2e-5}
                           ], lr=1e-3, weight_decay=0)


def lambda1(epoch): return (1 / 10)**epoch


def lambda2(epoch): return (1 / 10)**epoch


scheduler = T.optim.lr_scheduler.LambdaLR(optimizer, [lambda1, lambda2])


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

    # print(binary_logits.detach().cpu().numpy())

    binary_predictions = np.where(binary_logits.view(-1).detach().cpu().numpy() > 0.5, 1, 0)
    predictions = T.argmax(logits, dim=-1).detach().cpu().numpy()

    loss = utils.cross_entropy(model, logits, labels, binary_logits, binary_labels,
                               label_weights, binary_label_weights, label_mask)

    T.cuda.empty_cache()

    return predictions, binary_predictions, loss


epochs = 100

val_batches_texts, val_batches_text_ids, \
    val_batches_labels, val_batches_binary_labels, \
    val_batches_mask, val_batches_label_masks = bnb.bucket_and_batch(
        val_texts, val_labels, val_binary_labels,  labels2idx, binary_labels2idx, val_batch_size)

print("Validation batches loaded")

train_batches_texts, train_batches_text_ids, \
    train_batches_labels, train_batches_binary_labels, \
    train_batches_mask, train_batches_label_masks = bnb.bucket_and_batch(
        train_texts, train_labels, train_binary_labels, labels2idx, binary_labels2idx, train_batch_size)

print("Train batches loaded")

display_step = 100
example_display_step = 500
patience = 5

load = input("\nLoad checkpoint? y/n: ")
print("")

if load.lower() == 'y':

    print('Loading pre-trained weights for the model...')

    checkpoint = T.load("../Model_Backup/{}.pt".format(model_name))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    past_epoch = checkpoint['past epoch']
    best_val_F1 = checkpoint['best F1']
    best_val_cost = checkpoint['best loss']
    impatience = checkpoint['impatience']
    meta_impatience = checkpoint['meta_impatience']

    print('\nRESTORATION COMPLETE\n')

else:

    past_epoch = 0
    best_val_cost = math.inf
    best_val_F1 = -math.inf
    impatience = 0
    meta_impatience = 0

for epoch in range(past_epoch, epochs):

    batches_indices = [i for i in range(0, len(train_batches_texts))]
    random.shuffle(batches_indices)

    total_train_loss = 0
    total_F1 = 0

    for i in range(len(train_batches_texts)):

        j = int(batches_indices[i])

        # with T.autograd.detect_anomaly():

        predictions, binary_predictions, loss = predict(text_ids=train_batches_text_ids[j],
                                                        labels=train_batches_labels[j],
                                                        binary_labels=train_batches_binary_labels[j],
                                                        input_mask=train_batches_mask[j],
                                                        label_mask=train_batches_label_masks[j],
                                                        train=True)

        loss = loss/accu_step

        loss.backward()

        if (i+1) % accu_step == 0:
            # Update accumulated gradients
            T.nn.utils.clip_grad_norm_(parameters+BERT_parameters, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        labels = train_batches_labels[j].tolist()
        binary_labels = train_batches_binary_labels[j].tolist()
        predictions = predictions.tolist()
        binary_predictions = binary_predictions.tolist()
        label_masks = train_batches_label_masks[j].tolist()

        binary_prec, binary_rec, binary_acc = eval.binary_metrics(binary_predictions, binary_labels)
        prec, rec, acc = eval.multi_metrics(predictions, labels, label_masks, idx2labels)
        binary_F1 = eval.compute_F1(binary_prec, binary_rec)
        F1 = eval.compute_F1(prec, rec)

        cost = loss.item()

        if i % display_step == 0:

            print("Iter "+str(i)+", Cost = " +
                  "{:.3f}".format(cost)+", Binary F1 = " +
                  "{:.3f}".format(binary_F1)+", Multi-F1 = " +
                  "{:.3f}".format(F1)+", Binary Accuracy = " +
                  "{:.3f}".format(binary_acc)+", Accuracy = " +
                  "{:.3f}".format(acc))

        if i % example_display_step == 0:

            display(train_batches_texts[j],
                    predictions, labels,
                    binary_predictions, binary_labels,
                    label_masks)

    print("\n\n")

    total_val_cost = 0
    batch_labels = []
    batch_binary_labels = []
    batch_predictions = []
    batch_binary_predictions = []
    batch_label_masks = []

    for i in range(0, len(val_batches_texts)):

        if i % display_step == 0:
            print("Validating Batch {}".format(i+1))

        with T.no_grad():

            predictions, binary_predictions, loss = predict(text_ids=val_batches_text_ids[i],
                                                            labels=val_batches_labels[i],
                                                            binary_labels=val_batches_binary_labels[i],
                                                            input_mask=val_batches_mask[i],
                                                            label_mask=val_batches_label_masks[i],
                                                            train=False)

            cost = loss.item()

            total_val_cost += cost

            predictions = predictions.tolist()
            binary_predictions = binary_predictions.tolist()
            labels = val_batches_labels[i].tolist()
            binary_labels = val_batches_binary_labels[i].tolist()
            label_masks = val_batches_label_masks[i].tolist()

            batch_labels += labels
            batch_binary_labels += binary_labels
            batch_predictions += predictions
            batch_binary_predictions += binary_predictions
            batch_label_masks += label_masks

        if i % example_display_step == 0:

            display(val_batches_texts[i],
                    predictions, labels,
                    binary_predictions, binary_labels,
                    label_masks)

    binary_prec, binary_rec, binary_acc = eval.binary_metrics(batch_binary_predictions,
                                                              batch_binary_labels)
    prec, rec, acc = eval.multi_metrics(
        batch_predictions, batch_labels, batch_label_masks, idx2labels)
    binary_val_F1 = eval.compute_F1(binary_prec, binary_rec)
    val_F1 = eval.compute_F1(prec, rec)

    val_len = len(val_batches_texts)

    avg_val_cost = total_val_cost/val_len

    print("\n\nVALIDATION\n\n")

    print("Epoch "+str(epoch)+":, Cost = " +
          "{:.3f}".format(avg_val_cost)+", Binary F1 = " +
          "{:.3f}".format(binary_val_F1)+", Multi-F1 = " +
          "{:.3f}".format(val_F1)+", Binary Accuracy = " +
          "{:.3f}".format(binary_acc)+", Accuracy = " +
          "{:.3f}".format(acc))

    flag = 0
    impatience += 1

    if avg_val_cost < best_val_cost:

        impatience = 0

        best_val_cost = avg_val_cost

    if val_F1 >= best_val_F1:

        impatience = 0

        best_val_F1 = val_F1

        flag = 1

    if flag == 1:

        T.save({
            'past epoch': epoch+1,
            'best loss': best_val_cost,
            'best F1': best_val_F1,
            'impatience': impatience,
            'meta_impatience': meta_impatience,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, "../Model_Backup/{}.pt".format(model_name))

        print("Checkpoint created!")

    print("\n")

    if impatience > patience:
        scheduler.step()
        meta_impatience += 1
        if meta_impatience > patience:
            break
