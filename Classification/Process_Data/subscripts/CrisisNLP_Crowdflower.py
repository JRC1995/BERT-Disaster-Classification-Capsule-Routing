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

    directory = "../Data/CrisisNLP_Crowdflower/"

    data_dir = "../Processed_Data/Processed_Data_Intermediate.json"

    if not os.path.exists(data_dir):
        data = {}
    else:
        with open(data_dir) as file:
            data = json.load(file)

    disaster_keys = ["Pakistan_Earthquake", "California_Earthquake", "Chile_Earthquake",
                     "Ebola", "Hurricane_Odile", "India_Floods",
                     "Middle_East_Respiratory_Syndrome", "Pakistan_Floods", "Typhoon_Hagupit",
                     "Cyclone_Pam", "Nepal_Earthquake"]

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
                            if i > 0 and len(row) >= 3:
                                data[disaster_key]["tweet_ids"].append(row[0])
                                tweet = str(row[1])
                                tweet = tweet.encode("ascii", errors="ignore").decode()
                                data[disaster_key]["tweets"].append(tweet.lower())
                                data[disaster_key]["labels"].append(row[2])

    with open(data_dir, 'w') as outfile:
        json.dump(data, outfile)
