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

    directory = "../Data/CrisisMMD/"

    data_dir = "../Processed_Data/Processed_Data_Intermediate.json"

    if not os.path.exists(data_dir):
        data = {}
    else:
        with open(data_dir) as file:
            data = json.load(file)

    disaster_keys = ["California_Wildfires", "Hurricane_Harvey", "Hurricane_Irma",
                     "Hurricane_Maria", "Iraq_Iran_Earthquake", "Mexico_Earthquake",
                     "Srilanka_Floods"]

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

                    for key in disaster_keys:
                        if key.lower() in filename.lower():
                            disaster_key = key

                    if filename[-4:] == '.tsv':
                        encoding = 'ISO 8859-1'
                    else:
                        encoding = 'utf-8'

                    with codecs.open(filename, 'r', encoding=encoding) as csvfile:

                        if filename[-3:] == 'tsv':
                            csv_reader = csv.reader(csvfile, delimiter='\t')
                        else:
                            csv_reader = csv.reader(csvfile)

                        for i, row in enumerate(csv_reader):
                            if i > 0:
                                if len(row) > 12:
                                    data[disaster_key]["tweet_ids"].append(row[0])
                                    tweet = str(row[12])
                                    tweet = tweet.encode("ascii", errors="ignore").decode()
                                    if len(tweet.split(" ")) > 300:
                                        print("CrisisMMD: "+" ".join(tweet))
                                    data[disaster_key]["tweets"].append(tweet.lower())
                                    if row[6] == '':
                                        label = row[2]
                                        if row[2] == '':
                                            label = 'not_labeled'
                                    else:
                                        label = row[6]
                                    data[disaster_key]["labels"].append(label)

    with open(data_dir, 'w') as outfile:
        json.dump(data, outfile)
