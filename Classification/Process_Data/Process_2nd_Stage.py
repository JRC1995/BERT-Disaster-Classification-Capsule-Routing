import json
import re

data_dir = "../Processed_Data/Processed_Data_Intermediate.json"
with open(data_dir) as file:
    data = json.load(file)

all_tweet_ids = []
all_tweets = []
all_labels = []
all_disasters = []

unique_tweets = {}


def process_tweet(tweet):
    # pattern to match
    user_pattern = r'@[\w\.-]+[,\.;: ]+'
    # replace the matched pattern from string with,
    user_replace = r'@user '

    tweet = re.sub(user_pattern, user_replace, tweet)

    if tweet[0:3].lower() == "rt ":
        tweet = tweet[3:]

    tweet = tweet.split(" ")
    new_tweet = []
    for token in tweet:
        if 'https://' in token or 'http://' in token or 'www.' in token:
            pass
        else:
            new_tweet.append(token)

    tweet = " ".join(new_tweet)

    return tweet.lower()


id = 0
for disaster_key in data:
    tweet_ids = data[disaster_key]["tweet_ids"]
    tweets = data[disaster_key]["tweets"]
    labels = data[disaster_key]["labels"]

    for tweet_id, tweet, label in zip(tweet_ids, tweets, labels):
        tweet = process_tweet(tweet)
        tweet_id = str(tweet_id).strip("'")
        if tweet.strip() != '':
            if tweet not in unique_tweets:
                unique_tweets[tweet] = id
                all_tweet_ids.append(tweet_id)
                all_tweets.append(tweet)
                label = " ".join(label.split("\n"))
                all_labels.append([label])
                all_disasters.append([disaster_key])
                # print(tweet_id)
                # print(tweet)
                # print(label)
                id += 1
            else:
                id_ = unique_tweets[tweet]
                label = " ".join(label.split("\n"))
                all_labels[id_].append(label)
                all_disasters[id_].append(disaster_key)
                all_disasters[id_] = list(set(all_disasters[id_]))


# Analytics:

disaster_count = {}
label_count = {}
data_count = 0

for labels, disaster_keys in zip(all_labels, all_disasters):
    data_count += 1
    labels = list(set(labels))
    for label in labels:
        if label not in label_count:
            label_count[label] = 1
        else:
            label_count[label] += 1
    for key in disaster_keys:
        if key not in disaster_count:
            disaster_count[key] = 1
        else:
            disaster_count[key] += 1


file_data = "Total Data: {}".format(data_count)

file_data += "\n\nDisaster Statistics\n\n"

for key in disaster_count:
    name = key.split("_")
    name = " ".join(name)
    file_data += "{}: {}".format(name, disaster_count[key])+"\n"

file_data += "\n\nLabel Statistics\n\n"

for label in label_count:
    file_data += "{}: {}".format(label, label_count[label]) + "\n"

f = open("initial_stats.txt", "w")
f.write(file_data)
f.close()


d = {}
d["tweet_ids"] = all_tweet_ids
d["tweets"] = all_tweets
d["labels"] = all_labels
d['disasters'] = all_disasters

with open("../Processed_Data/Processed_Data_Intermediate_Stage_2.json", 'w') as outfile:
    json.dump(d, outfile)
