import numpy as np
import re
import random
import string
import math
import csv
from langdetect import detect
import codecs
import os
import json


def process():

    directory = "../Data/BigCrisisData/"

    data_dir = "../Processed_Data/Processed_Data_Intermediate.json"

    if not os.path.exists(data_dir):
        data = {}
    else:
        with open(data_dir) as file:
            data = json.load(file)

    disaster_keys = ["BigCrisis"]

    for disaster_key in disaster_keys:

        if disaster_key not in data:
            data[disaster_key] = {}

        if "tweet_ids" not in data[disaster_key]:
            data[disaster_key]["tweet_ids"] = []

        if "tweets" not in data[disaster_key]:
            data[disaster_key]["tweets"] = []

        if "labels" not in data[disaster_key]:
            data[disaster_key]["labels"] = []

    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            filename = os.path.join(root, file)

            if len(filename) > 4:
                if filename[-4:] == '.tsv' or filename[-4:] == '.csv':

                    print(filename)

                    if filename[-4:] == '.tsv':
                        encoding = 'ISO 8859-1'
                    else:
                        encoding = 'utf-8'

                    with codecs.open(filename, 'r', encoding=encoding) as csvfile:

                        disaster_key = "BigCrisis"

                        if filename[-3:] == 'tsv':
                            csv_reader = csv.reader(csvfile, delimiter='\t')
                        else:
                            csv_reader = csv.reader(csvfile)

                        for i, row in enumerate(csv_reader):
                            if i > 0:
                                data[disaster_key]["tweet_ids"].append(row[0])
                                tweet = str(row[1])
                                tweet = tweet.encode("ascii", errors="ignore").decode()
                                data[disaster_key]["tweets"].append(tweet.lower())
                                data[disaster_key]["labels"].append(row[2])

    """
    display_step = 100

    for disaster_key in disaster_keys:

        tweet_ids = data[disaster_key]["tweet_ids"]

        tweets = data[disaster_key]["tweets"]

        labels = data[disaster_key]["labels"]

        print("\n\n{}\n\n".format(disaster_key))

        i = 0

        for tweet_id, tweet, label in zip(tweet_ids, tweets, labels):
            if i % display_step == 0:
                print("tweet_id: ", tweet_id)
                print("tweet: ", tweet)
                print("label: ", label)
                print("\n\n")
    """

    with open(data_dir, 'w') as outfile:
        json.dump(data, outfile)
