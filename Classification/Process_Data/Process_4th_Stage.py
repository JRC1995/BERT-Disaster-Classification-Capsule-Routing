import json
import re
import random

data_dir = "../Processed_Data/Processed_Data_Intermediate_Stage_3.json"
with open(data_dir) as file:
    data = json.load(file)

tweet_ids = data['tweet_ids']
tweets = data['tweets']
labels = data['labels']
binary_labels = data['binary_labels']
#all_disasters = data['disasters']

random.seed(101)

randomized_indices = [i for i in range(len(tweet_ids))]
random.shuffle(randomized_indices)

tweet_ids = [tweet_ids[i] for i in randomized_indices]
tweets = [tweets[i] for i in randomized_indices]
labels = [labels[i] for i in randomized_indices]
binary_labels = [binary_labels[i] for i in randomized_indices]

test_len = 20000
val_len = 10000

test_tweet_ids = tweet_ids[0:test_len]
test_tweets = tweets[0:test_len]
test_labels = labels[0:test_len]
test_binary_labels = binary_labels[0:test_len]

val_tweet_ids = tweet_ids[test_len:val_len+test_len]
val_tweets = tweets[test_len:val_len+test_len]
val_labels = labels[test_len:val_len+test_len]
val_binary_labels = binary_labels[test_len:val_len+test_len]

train_tweet_ids = tweet_ids[val_len+test_len:]
train_tweets = tweets[val_len+test_len:]
train_labels = labels[val_len+test_len:]
train_binary_labels = binary_labels[val_len+test_len:]

print("\n\nExample Training Data\n\n")
for i in range(10):
    print("ID:", train_tweet_ids[i])
    print("Tweet:", train_tweets[i])
    print("Label:", train_labels[i])
    print("Binary Label:", train_binary_labels[i])
    print("\n")

print("\n\nExample Validation Data\n\n")
for i in range(10):
    print("ID:", val_tweet_ids[i])
    print("Tweet:", val_tweets[i])
    print("Label:", val_labels[i])
    print("Binary Label:", val_binary_labels[i])
    print("\n")

print("\n\nExample Testing Data\n\n")
for i in range(10):
    print("ID:", test_tweet_ids[i])
    print("Tweet:", test_tweets[i])
    print("Label:", test_labels[i])
    print("Binary Label:", test_binary_labels[i])
    print("\n")

file_data = "Training Data: {}".format(len(train_tweet_ids))+"\n"
file_data += "Validation Data: {}".format(len(val_tweet_ids))+"\n"
file_data += "Testing Data: {}".format(len(test_tweet_ids))+"\n"

f = open("split_stats.txt", "w")
f.write(file_data)
f.close()

d = {}
d["tweet_ids"] = train_tweet_ids
d["tweets"] = train_tweets
d["labels"] = train_labels
d["binary_labels"] = train_binary_labels

with open("../Processed_Data/train_data.json", 'w') as outfile:
    json.dump(d, outfile)

d = {}
d["tweet_ids"] = val_tweet_ids
d["tweets"] = val_tweets
d["labels"] = val_labels
d["binary_labels"] = val_binary_labels

with open("../Processed_Data/val_data.json", 'w') as outfile:
    json.dump(d, outfile)

d = {}
d["tweet_ids"] = test_tweet_ids
d["tweets"] = test_tweets
d["labels"] = test_labels
d["binary_labels"] = train_binary_labels

with open("../Processed_Data/test_data.json", 'w') as outfile:
    json.dump(d, outfile)


d = {}
d["tweet_ids"] = train_tweet_ids
d["labels"] = train_labels
d["binary_labels"] = train_binary_labels

with open("../Processed_Data/train_data_public.json", 'w') as outfile:
    json.dump(d, outfile)

d = {}
d["tweet_ids"] = val_tweet_ids
d["labels"] = val_labels
d["binary_labels"] = val_binary_labels

with open("../Processed_Data/val_data_public.json", 'w') as outfile:
    json.dump(d, outfile)

d = {}
d["tweet_ids"] = test_tweet_ids
d["labels"] = test_labels
d["binary_labels"] = train_binary_labels

with open("../Processed_Data/test_data_public.json", 'w') as outfile:
    json.dump(d, outfile)
