import numpy as np
import random
import re
import pickle
from transformers import *
import copy


class bucket_and_batch:

    def bucket_and_batch(self, texts, labels, binary_labels,
                         labels2idx, binary_labels2idx, batch_size, model='BERT'):

        idx2labels = {v: k for k, v in labels2idx.items()}
        binary_idx2labels = {v: k for k, v in binary_labels2idx.items()}

        if model.lower() == 'bert':

            tokenizer = BertTokenizer.from_pretrained('../Pre_trained_BERT/')

            PAD = tokenizer.encode("[PAD]")[0]

        else:

            tokenizer = XLMTokenizer.from_pretrained('../Pre_trained_XLM/')

            PAD = tokenizer.encode("<pad>")[0]

        original_texts = copy.deepcopy(texts)
        text_ids = [tokenizer.encode(text, add_special_tokens=True,
                                     max_length=500) for text in texts]

        true_seq_lens = np.zeros((len(texts)), dtype=int)

        for i in range(len(text_ids)):
            true_seq_lens[i] = len(text_ids[i])

        # sorted in descending order after flip
        sorted_by_len_indices = np.flip(np.argsort(true_seq_lens), 0)

        sorted_texts = []
        sorted_text_ids = []
        sorted_labels = []
        sorted_binary_labels = []

        for i in range(len(texts)):

            text_id = text_ids[sorted_by_len_indices[i]]
            # if len(text_id) > 500:
            #text_id = text_id[0:500]

            sorted_texts.append(
                original_texts[sorted_by_len_indices[i]])

            sorted_text_ids.append(text_id)

            sorted_labels.append(
                labels[sorted_by_len_indices[i]])

            sorted_binary_labels.append(
                binary_labels[sorted_by_len_indices[i]])

        print("Sample size: ", len(sorted_texts))

        i = 0
        batches_texts = []
        batches_text_ids = []
        batches_labels = []
        batches_binary_labels = []
        batches_mask = []
        batches_label_masks = []
        count = 0

        while i < len(sorted_texts):

            if i+batch_size > len(sorted_texts):
                batch_size = len(sorted_texts)-i

            batch_texts = []
            batch_text_ids = []
            batch_labels = []
            batch_binary_labels = []
            batch_mask = []
            batch_label_masks = []

            max_len = len(sorted_text_ids[i])

            for j in range(i, i + batch_size):

                text = sorted_texts[j]
                text_id = sorted_text_ids[j]
                attention_mask = [1 for k in range(len(text_id))]
                label = labels2idx.get(sorted_labels[j], labels2idx['not informative'])
                binary_label = binary_labels2idx[sorted_binary_labels[j]]

                if sorted_labels[j] == 'unlabeled':
                    label_mask = 0
                else:
                    label_mask = 1

                while len(text_id) < max_len:
                    text_id.append(PAD)
                    attention_mask.append(0)

                # print(text)
                # print(label_mask)
                #print("multi:", idx2labels[label])
                #print("binary:", binary_idx2labels[binary_label])

                batch_texts.append(text)
                batch_text_ids.append(text_id)
                batch_labels.append(label)
                batch_binary_labels.append(binary_label)
                batch_mask.append(attention_mask)
                batch_label_masks.append(label_mask)

            batch_text_ids = np.asarray(batch_text_ids, dtype=int)
            batch_labels = np.asarray(batch_labels, dtype=int)
            batch_binary_labels = np.asarray(batch_binary_labels, dtype=int)
            batch_mask = np.asarray(batch_mask, dtype=int)
            batch_label_masks = np.asarray(batch_label_masks, dtype=int)

            batches_texts.append(batch_texts)
            batches_text_ids.append(batch_text_ids)
            batches_labels.append(batch_labels)
            batches_binary_labels.append(batch_binary_labels)
            batches_mask.append(batch_mask)
            batches_label_masks.append(batch_label_masks)

            i += batch_size

        return batches_texts, batches_text_ids, batches_labels,\
            batches_binary_labels, batches_mask, batches_label_masks
