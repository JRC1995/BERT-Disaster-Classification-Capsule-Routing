import numpy as np


def binary_metrics(predictions, labels):
    tp = 0
    correct = 0
    pred_len = 0
    gold_len = 0
    total = 0
    for prediction, label in zip(predictions, labels):
        if prediction == label:
            correct += 1
            if prediction == 1:
                tp += 1
        if prediction == 1:
            pred_len += 1
        if label == 1:
            gold_len += 1
        total += 1
    precision = tp/pred_len if pred_len > 0 else 0
    recall = tp/gold_len if gold_len > 0 else 0
    accuracy = correct/total
    return precision, recall, accuracy


def multi_metrics(predictions, labels, label_masks, idx2labels, verbose=False):

    total_precision = 0
    total_recall = 0

    for id in idx2labels:
        tp = 0
        correct = 0
        pred_len = 0
        gold_len = 0
        total = 0
        for prediction, label, label_mask in zip(predictions, labels, label_masks):
            if label_mask == 1:
                if prediction == label:
                    correct += 1
                    if prediction == id:
                        tp += 1
                if prediction == id:
                    pred_len += 1
                if label == id:
                    gold_len += 1
                total += 1
        precision = tp/pred_len if pred_len > 0 else 0
        recall = tp/gold_len if gold_len > 0 else 0
        accuracy = correct/total if total > 0 else 0

        if verbose:

            print("id", id)
            print("pred_len", pred_len)
            print("gold_len", gold_len)
            print("precision", precision)
            print("recall", recall)
            print("\n\n")

        total_precision += precision
        total_recall += recall

    precision = total_precision/len(idx2labels)
    recall = total_recall/len(idx2labels)

    return precision, recall, accuracy


def compute_F1(precision, recall):
    F1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
    return F1
