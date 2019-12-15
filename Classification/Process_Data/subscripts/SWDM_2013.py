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

    directory = "../Data/SWDM_2013/"

    data_dir = "../Processed_Data/Processed_Data_Intermediate.json"

    if not os.path.exists(data_dir):
        data = {}
    else:
        with open(data_dir) as file:
            data = json.load(file)

    disaster_keys = ["Hurricane_Sandy"]

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

                    disaster_key = "Hurricane_Sandy"

                    encoding = 'ISO 8859-1'

                    with codecs.open(filename, 'r', encoding=encoding) as csvfile:

                        if filename[-3:] == 'tsv':
                            csv_reader = csv.reader(csvfile, delimiter='\t')
                        else:
                            csv_reader = csv.reader(csvfile)

                        for i, row in enumerate(csv_reader):
                            if i > 0 and len(row) >= 11:
                                label = row[7]
                                if label == '':
                                    label = row[5]
                                if label != '':
                                    data[disaster_key]["tweet_ids"].append(row[0])
                                    if filename == '../Data/SWDM_2013/SWDM2013/sandy2012_labeled_data/01_personal-informative-other/a143145.csv':
                                        num = 10
                                    else:
                                        # for j in range(len(row)):
                                            #print("j: {}, rowj: {}".format(j, row[j]))
                                        # print("\n")
                                        num = 9
                                    tweet = str(row[num])
                                    tweet = tweet.encode("ascii", errors="ignore").decode()
                                    if len(tweet.split(" ")) > 300:
                                        print("SWDM: "+" ".join(tweet))
                                    data[disaster_key]["tweets"].append(tweet.lower())
                                    data[disaster_key]["labels"].append(row[5])

    with open(data_dir, 'w') as outfile:
        json.dump(data, outfile)
