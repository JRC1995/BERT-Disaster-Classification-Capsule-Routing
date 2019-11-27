import json
import re

label_map = {'Other Useful Information': 'unlabeled',
             'other_useful_information': 'unlabeled',
             'relevant': 'unlabeled',
             'Infrastructure and utilities': 'casaulties and damage',
             'Informative (Direct or Indirect)': 'unlabeled',
             'Informative (Indirect)': 'unlabeled',
             'Informative (Direct)': 'unlabeled',
             'Casualties and damage': 'casaulties and damage',
             'Affected individuals': 'casaulties and damage',
             'injured_or_dead_people': 'casaulties and damage',
             'not informative': 'not informative',
             'not_informative': 'not informative',
             'other_relevant_information': 'unlabeled',
             'infrastructure_and_utilities_damage': 'casaulties and damage',
             'Not applicable': 'unlabeled',
             'Not labeled': 'unlabeled',
             'Sympathy and support': 'not informative',
             'rescue_volunteering_or_donation_effort': 'donation',
             'Information Source': 'unlabeled',
             'Caution and advice': 'caution and advice',
             'affected_individuals': 'casaulties and damage',
             'Not Relevant': 'not informative',
             'not_relevant': 'not informative',
             'infrastructure_and_utility_damage': 'casaulties and damage',
             'donation_needs_or_offers_or_volunteering_services': 'donation',
             'caution_and_advice': 'caution and advice',
             'Not related or irrelevant': 'not informative',
             'on-topic': 'unlabeled',
             'Not Informative': 'not informative',
             'not_related_or_irrelevant': 'not informative',
             'Not related to crisis': 'not informative',
             'Not informative': 'not informative',
             'Not physical landslide': 'not informative',
             'Unknown': 'unlabeled',
             'Donations and volunteering': 'donation',
             'off-topic': 'not informative',
             'Informative': 'unlabeled',
             'Donations of money, goods or services': 'donation',
             'Informative(Indirect)': 'unlabeled',
             'not_relevant_or_cant_judge': 'unlabeled',
             'Informative(Direct)': 'unlabeled',
             'sympathy_and_emotional_support': 'not informative',
             'Physical landslide': 'unlabeled',
             'Other': 'unlabeled',
             'Personal Only': 'not informative',
             'Injured and dead': 'casaulties and damage',
             'Donations of supplies and/or volunteer work': 'donation',
             'treatment': 'casaulties and damage',
             'Other useful information': 'unlabeled',
             'No': 'unlabeled',
             'Personal only': 'not informative',
             'Other Relevant Information': 'unlabeled',
             'Yes': 'unlabeled',
             'Praying': 'not informative',
             'Needs of those affected': 'casaulties and damage',
             'vehicle_damage': 'casaulties and damage',
             'missing_trapped_or_found_people': 'casaulties and damage',
             'prevention': 'caution and advice',
             'Personal updates, sympathy, support': 'not informative',
             'disease_transmission': 'unlabeled',
             '': 'unlabeled',
             'Personal updates': 'not informative',
             'Infrastructure': 'casaulties and damage',
             'Other relevant information': 'unlabeled',
             'Sympathy and emotional support': 'not informative',
             'Volunteer or professional services': 'donation',
             'displaced_people_and_evacuations': 'casaulties and damage',
             'Money': 'donation',
             'Infrastructure Damage': 'casaulties and damage',
             'Humanitarian Aid Provided': 'donation',
             'Displaced people': 'casaulties and damage',
             'Injured or dead people': 'casaulties and damage',
             'affected_people': 'casaulties and damage',
             'disease_signs_or_symptoms': 'caution and advice',
             'Requests for Help/Needs': 'casaulties and damage',
             'Urgent Needs': 'casaulties and damage',
             'Shelter and supplies': 'unlabeled',
             'Response Efforts': 'casaulties and damage',
             'Missing, trapped, or found people': 'casaulties and damage',
             'Donations of money': 'donation',
             'missing_or_found_people': 'casaulties and damage',
             'dont_know_or_cant_judge': 'unlabeled',
             'People missing or found': 'casaulties and damage',
             'deaths_reports': 'casaulties and damage',
             'Other relevant': 'unlabeled',
             'Not relevant': 'not informative',
             'Response efforts': 'casaulties and damage',
             'Infrastructure damage': 'casaulties and damage',
             'Animal management': 'unlabeled',
             'Non-government': 'unlabeled',
             'Personal': 'not informative',
             'Traditional media': 'unlabeled',
             'Informative(Direct) Informative(Direct or Indirect)': 'unlabeled',
             'Informative (Direct) Informative (Direct or Indirect)': 'unlabeled',
             'Personal only Other': 'not informative',
             'Informative(Indirect) Informative(Direct or Indirect)': 'unlabeled',
             'Informative (Indirect) Informative (Direct or Indirect)': 'unlabeled',
             'Caution and advice Unknown': 'unlabeled',
             'People missing, found or seen': 'casaulties and damage',
             'Information source': 'unlabeled',
             'Caution and advice Casualties and damage Unknown': 'unlabeled',
             'Casualties and damage Information source': 'unlabeled',
             'Caution and advice Casualties and damage Information source': 'unlabeled',
             'Caution and advice Information source': 'unlabeled'}


binary_label_map = {'Other Useful Information': 'informative',
                    'other_useful_information': 'informative',
                    'relevant': 'informative',
                    'Infrastructure and utilities': 'informative',
                    'Informative (Direct or Indirect)': 'informative',
                    'Informative (Direct)': 'informative',
                    'Informative (Indirect)': 'informative',
                    'Casualties and damage': 'informative',
                    'Affected individuals': 'informative',
                    'injured_or_dead_people': 'informative',
                    'not informative': 'not informative',
                    'not_informative': 'not informative',
                    'other_relevant_information': 'informative',
                    'infrastructure_and_utilities_damage': 'informative',
                    'Not applicable': 'unlabeled',
                    'Not labeled': 'unlabeled',
                    'Sympathy and support': 'not informative',
                    'rescue_volunteering_or_donation_effort': 'informative',
                    'Information Source': 'unlabeled',
                    'Caution and advice': 'informative',
                    'affected_individuals': 'informative',
                    'Not Relevant': 'not informative',
                    'not_relevant': 'not informative',
                    'infrastructure_and_utility_damage': 'informative',
                    'donation_needs_or_offers_or_volunteering_services': 'informative',
                    'caution_and_advice': 'informative',
                    'Not related or irrelevant': 'not informative',
                    'on-topic': 'informative',
                    'Not Informative': 'not informative',
                    'not_related_or_irrelevant': 'not informative',
                    'Not related to crisis': 'not informative',
                    'Not informative': 'not informative',
                    'Not physical landslide': 'not informative',
                    'Unknown': 'unlabeled',
                    'Donations and volunteering': 'informative',
                    'off-topic': 'not informative',
                    'Informative': 'informative',
                    'Donations of money, goods or services': 'informative',
                    'Informative(Indirect)': 'informative',
                    'not_relevant_or_cant_judge': 'unlabeled',
                    'Informative(Direct)': 'informative',
                    'sympathy_and_emotional_support': 'not informative',
                    'Physical landslide': 'informative',
                    'Other': 'informative',
                    'Personal Only': 'not informative',
                    'Injured and dead': 'informative',
                    'Donations of supplies and/or volunteer work': 'informative',
                    'treatment': 'informative',
                    'Other useful information': 'informative',
                    'No': 'unlabeled',
                    'Personal only': 'not informative',
                    'Other Relevant Information': 'informative',
                    'Yes': 'unlabeled',
                    'Praying': 'not informative',
                    'Needs of those affected': 'informative',
                    'vehicle_damage': 'informative',
                    'missing_trapped_or_found_people': 'informative',
                    'prevention': 'informative',
                    'Personal updates, sympathy, support': 'not informative',
                    'disease_transmission': 'informative',
                    '': 'unlabeled',
                    'Personal updates': 'not informative',
                    'Infrastructure': 'informative',
                    'Other relevant information': 'informative',
                    'Sympathy and emotional support': 'not informative',
                    'Volunteer or professional services': 'informative',
                    'displaced_people_and_evacuations': 'informative',
                    'Money': 'informative',
                    'Infrastructure Damage': 'informative',
                    'Humanitarian Aid Provided': 'informative',
                    'Displaced people': 'informative',
                    'Injured or dead people': 'informative',
                    'affected_people': 'informative',
                    'disease_signs_or_symptoms': 'informative',
                    'Requests for Help/Needs': 'informative',
                    'Urgent Needs': 'informative',
                    'Shelter and supplies': 'informative',
                    'Response Efforts': 'informative',
                    'Missing, trapped, or found people': 'informative',
                    'Donations of money': 'informative',
                    'missing_or_found_people': 'informative',
                    'dont_know_or_cant_judge': 'unlabeled',
                    'People missing or found': 'informative',
                    'deaths_reports': 'informative',
                    'Other relevant': 'informative',
                    'Not relevant': 'not informative',
                    'Response efforts': 'informative',
                    'Infrastructure damage': 'informative',
                    'Animal management': 'informative',
                    'Non-government': 'unlabeled',
                    'Personal': 'not informative',
                    'Traditional media': 'unlabeled',
                    'Informative(Direct) Informative(Direct or Indirect)': 'unlabeled',
                    'Informative (Direct) Informative (Direct or Indirect)': 'unlabeled',
                    'Personal only Other': 'not informative',
                    'Informative(Indirect) Informative(Direct or Indirect)': 'unlabeled',
                    'Informative (Indirect) Informative (Direct or Indirect)': 'unlabeled',
                    'Caution and advice Unknown': 'unlabeled',
                    'People missing, found or seen': 'informative',
                    'Information source': 'unlabeled',
                    'Caution and advice Casualties and damage Unknown': 'unlabeled',
                    'Casualties and damage Information source': 'unlabeled',
                    'Caution and advice Casualties and damage Information source': 'unlabeled',
                    'Caution and advice Information source': 'unlabeled'}

label_rank = {'casaulties and damage': 0,
              'caution and advice': 1,
              'donation': 2,
              'not informative': 3,
              'unlabeled': 4}

binary_label_rank = {'informative': 0,
                     'not informative': 1,
                     'unlabeled': 2}


data_dir = "../Processed_Data/Processed_Data_Intermediate_Stage_2.json"
with open(data_dir) as file:
    data = json.load(file)

tweet_ids = data['tweet_ids']
tweets = data['tweets']
all_labels = data['labels']
all_disasters = data['disasters']

new_tweet_ids = []
new_tweets = []
new_labels = []
new_binary_labels = []
new_all_disasters = []

for tweet_id, tweet, labels, disaster_keys in zip(tweet_ids, tweets, all_labels, all_disasters):
    labels = list(set(labels))
    reduced_labels = [label_map[label] for label in labels]
    label_scores = [label_rank[label] for label in reduced_labels]
    label = reduced_labels[label_scores.index(min(label_scores))]

    binary_labels = [binary_label_map[label] for label in labels]
    binary_label_scores = [binary_label_rank[label] for label in binary_labels]
    binary_label = binary_labels[binary_label_scores.index(min(binary_label_scores))]

    if binary_label != 'not informative' and label == 'not informative':
        label = 'unlabeled'

    if binary_label == 'unlabeled' and label == 'unlabeled':
        pass
    else:
        new_tweet_ids.append(tweet_id)
        new_tweets.append(tweet)
        new_labels.append(label)
        new_binary_labels.append(binary_label)
        new_all_disasters.append(disaster_keys)


# Analytics:

disaster_count = {}
label_count = {}
binary_label_count = {}
data_count = 0

for label, binary_label, disaster_keys in zip(new_labels, new_binary_labels, new_all_disasters):

    data_count += 1

    if label not in label_count:
        label_count[label] = 1
    else:
        label_count[label] += 1

    if binary_label not in binary_label_count:
        binary_label_count[binary_label] = 1
    else:
        binary_label_count[binary_label] += 1

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

file_data += "\n\nBinary Label Statistics\n\n"

for binary_label in binary_label_count:
    file_data += "{}: {}".format(binary_label, binary_label_count[binary_label]) + "\n"

file_data += "\n\nLabel Statistics\n\n"

for label in label_count:
    file_data += "{}: {}".format(label, label_count[label]) + "\n"

f = open("final_stats.txt", "w")
f.write(file_data)
f.close()

labels2idx = {'casaulties and damage': 0,
              'caution and advice': 1,
              'donation': 2,
              'not informative': 3}

binary_labels2idx = {'informative': 1,
                     'not informative': 0}

binary_label_counts = [v for k, v in binary_label_count.items()]
binary_label_weights = {k: max(binary_label_counts)/v for k, v in binary_label_count.items()}

label_counts = [v for k, v in label_count.items() if k != "unlabeled"]
label_weights = {k: max(label_counts)/v for k, v in label_count.items()}

d = {}
d["labels2idx"] = labels2idx
d["binary_labels2idx"] = binary_labels2idx
d["label_weights"] = label_weights
d["binary_label_weights"] = binary_label_weights

with open("../Processed_Data/label_info.json", 'w') as outfile:
    json.dump(d, outfile)

d = {}
d["tweet_ids"] = new_tweet_ids
d["tweets"] = new_tweets
d["labels"] = new_labels
d["binary_labels"] = new_binary_labels
d['disasters'] = new_all_disasters

with open("../Processed_Data/Processed_Data_Intermediate_Stage_3.json", 'w') as outfile:
    json.dump(d, outfile)
