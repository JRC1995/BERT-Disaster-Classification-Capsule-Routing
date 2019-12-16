# Disaster-related Tweet Classification with BERT
#### Exploring Attention-based and Capsule-Routing-based Layer Aggregation and Hidden state Aggregation

See Project Report [here](https://drive.google.com/file/d/101NgTXm7zicqhrxr9e6n3bWPBLzFK50o/view?usp=sharing) and Presentation
Slides [here](https://docs.google.com/presentation/d/1R8PV1bEQqgApC2vYTMQhP0pXRvDjZ5qXnsOaLhyB31w/edit?usp=sharing)

## Main Requirements

* Numpy
* Pytorch 1.3
* HuggingFace's [Transformers](https://github.com/huggingface/transformers) v2.2.2

## Credits

Besides HuggingFace's library we used [Heinsen Routing official code repository](https://github.com/glassroom/heinsen_routing) as a reference.

## Related Work

See [here](http://web.stanford.edu/class/cs224n/reports/custom/15785631.pdf) and [here](https://github.com/sebsk/CS224N-Project).

## Datasets Used

We collected our data from various sources:

* [Resource # 1 from CrisisNLP](https://crisisnlp.qcri.org/lrec2016/lrec2016.html)
```
@InProceedings{imran2016lrec,
  author = {Imran, Muhammad and Mitra, Prasenjit and Castillo, Carlos},
  title = {Twitter as a Lifeline: Human-annotated Twitter Corpora for NLP of Crisis-related Messages},
  booktitle = {Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC 2016)},
  year = {2016},
  month = {may},
  date = {23-28},
  location = {Portoroz, Slovenia},
  publisher = {European Language Resources Association (ELRA)},
  address = {Paris, France},
  isbn = {978-2-9517408-9-1},
  language = {english}
 }

```
* [Resource # 2 from CrisisNLP](https://crisisnlp.qcri.org/)
```
@inproceedings{imran2013practical,
title={Practical extraction of disaster-relevant information from social media},
author={Imran, Muhammad and Elbassuoni, Shady and Castillo, Carlos and Diaz, Fernando and Meier, Patrick},
booktitle={Proceedings of the 22nd international conference on World Wide Web companion},
pages={1021--1024},
year={2013},
organization={International World Wide Web Conferences Steering Committee}
}
```
* [Resource # 3 from CrisisNLP](https://crisisnlp.qcri.org/)
```
@article{imran2013extracting,
title={Extracting information nuggets from disaster-related messages in social media},
author={Imran, Muhammad and Elbassuoni, Shady Mamoon and Castillo, Carlos and Diaz, Fernando and Meier, Patrick},
journal={Proc. of ISCRAM, Baden-Baden, Germany},
year={2013}
}
```
* [Resource # 4 from CrisisNLP](https://github.com/CrisisNLP/deep-learning-for-big-crisis-data)
```
@inproceedings{nguyen2017robust,
  title={Robust classification of crisis-related data on social networks using convolutional neural networks},
  author={Nguyen, Dat Tien and Al Mannai, Kamela Ali and Joty, Shafiq and Sajjad, Hassan and Imran, Muhammad and Mitra, Prasenjit},
  booktitle={Eleventh International AAAI Conference on Web and Social Media},
  year={2017}
}
```
* [Resource # 5 from CrisisNLP](https://crisisnlp.qcri.org/)
```
@InProceedings{crisismmd2018icwsm,
  author = {Alam, Firoj and Ofli, Ferda and Imran, Muhammad},
  title = { CrisisMMD: Multimodal Twitter Datasets from Natural Disasters},
  booktitle = {Proceedings of the 12th International AAAI Conference on Web and Social Media (ICWSM)},
  year = {2018},
  month = {June},
  date = {23-28},
  location = {USA}
}
```
* [Resource # 7 from CrisisNLP](https://crisisnlp.qcri.org/)
```
@inproceedings{firoj_ACL_2018embaddings,
title={Domain Adaptation with Adversarial Training and Graph Embeddings},
author={Alam, Firoj and Joty, Shafiq and Imran, Muhammad},
journal={Proc. of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
year={2018}
}
```
* [Resource # 10 from CrisisNLP](https://crisisnlp.qcri.org/)
```
@inproceedings{alam2018graph,
  title={Graph Based Semi-Supervised Learning with Convolution Neural Networks to Classify Crisis Related Tweets},
  author={Alam, Firoj and Joty, Shafiq and Imran, Muhammad},
  booktitle={Twelfth International AAAI Conference on Web and Social Media},
  year={2018}
}
```
* [CrisisLexT26](https://www.crisislex.org/data-collections.html#CrisisLexT26)
```
[Olteanu et al. 2015] Alexandra Olteanu, Sarah Vieweg, Carlos Castillo. 2015. What to Expect When the Unexpected Happens: Social Media Communications Across Crises. In Proceedings of the ACM 2015 Conference on Computer Supported Cooperative Work and Social Computing (CSCW '15). ACM, Vancouver, BC, Canada.
```
* [CrisisLexT6](https://www.crisislex.org/data-collections.html#CrisisLexT6)
```
[Olteanu et al. 2014] Alexandra Olteanu, Carlos Castillo, Fernando Diaz, Sarah Vieweg: "CrisisLex: A Lexicon for Collecting and Filtering Microblogged Communications in Crises". ICWSM 2014.
```
## Dataset Statistics

See [here](https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/blob/master/Classification/Process_Data/final_stats.txt).

We prepare annotations for both multi-class classification and binary classification. We train for both task in a multi-task framework.

## Label Reduction Scheme

See this [code](https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/blob/master/Classification/Process_Data/Process_3rd_Stage.py) for the exact label reduction scheme. 


## Data Processing

First, the data can be either downloaded from the above links or collected from the tweet ids I make publicly available:
https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/tree/master/Classification/Processed_Data

However, some of the tweet ids are not exactly tweet ids but some other ids (especially for the data from Resource #2 and Resource #3). You can get the same data by downloading from resource #2 and # 3 or you can also just filter them out (check for the length of tweet ids - their length will be much less) and ignore them. Should not make a lot of difference.

There are also some issues with extracting data in our codes from resource #2 and #3. Multiple rows get concatenated into one string. I noticed it too late, and although I could try to fix it, I didn't because one 1 or 2 samples were like this. 

Now if you download from tweet ids, they are already annotated in the json files. If you load the JSON file in some variable "json_data" then you can access the list of tweet ids from:
```
json_data["tweet_ids"]
```
You can access the corresponding multi-class labels from:
```
json_data["labels"]
```
You can access the corresponding binary labels from:
```
json_data["binary_labels"]
```
But you need to pre-process the tweets yourself. For pre-processing use the **process_tweet** function from [here](https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/blob/master/Classification/Process_Data/Process_2nd_Stage.py).

Instead of downloading from tweet ids, if you download from the above mentioned links then you can run the pre-processing files sequentially to process the data and split them into training, validation, and testing sets. Precisely, you have run the following files in the specific order:

1. [Process_1st_Stage.py](https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/blob/master/Classification/Process_Data/Process_1st_Stage.py)

2. [Process_2nd_Stage.py](https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/blob/master/Classification/Process_Data/Process_2nd_Stage.py)

3. [Process_3rd_Stage.py](https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/blob/master/Classification/Process_Data/Process_3rd_Stage.py)

4. [Process_4th_Stage.py](https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/blob/master/Classification/Process_Data/Process_4th_Stage.py)

There are surely better ways to set up the pre-processing part than running them manually in this manner, but I am a bad coder. 

However, **BEFORE** you run the pre-processing files you have to take care of two things: Directories and Filenames.

This is the exact directory structure that I used (all the data should be put [here](https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/tree/master/Classification/Data)):

```
├── BigCrisisData
│   ├── sample.csv
│   ├── sample_prccd.csv
│   ├── sample_prccd_dev.csv
│   ├── sample_prccd_test.csv
│   └── sample_prccd_train.csv
├── CrisisLexT26
│   ├── 2012_Colorado_wildfires
│   │   ├── 2012_Colorado_wildfires-event_description.json
│   │   ├── 2012_Colorado_wildfires-tweetids_entire_period.csv
│   │   ├── 2012_Colorado_wildfires-tweets_labeled.csv
│   │   └── README.md
│   ├── 2012_Costa_Rica_earthquake
│   │   ├── 2012_Costa_Rica_earthquake-event_description.json
│   │   ├── 2012_Costa_Rica_earthquake-tweetids_entire_period.csv
│   │   ├── 2012_Costa_Rica_earthquake-tweets_labeled.csv
│   │   └── README.md
│   ├── 2012_Guatemala_earthquake
│   │   ├── 2012_Guatemala_earthquake-event_description.json
│   │   ├── 2012_Guatemala_earthquake-tweetids_entire_period.csv
│   │   ├── 2012_Guatemala_earthquake-tweets_labeled.csv
│   │   └── README.md
│   ├── 2012_Italy_earthquakes
│   │   ├── 2012_Italy_earthquakes-event_description.json
│   │   ├── 2012_Italy_earthquakes-tweetids_entire_period.csv
│   │   ├── 2012_Italy_earthquakes-tweets_labeled.csv
│   │   └── README.md
│   ├── 2012_Philipinnes_floods
│   │   ├── 2012_Philipinnes_floods-event_description.json
│   │   ├── 2012_Philipinnes_floods-tweetids_entire_period.csv
│   │   ├── 2012_Philipinnes_floods-tweets_labeled.csv
│   │   └── README.md
│   ├── 2012_Typhoon_Pablo
│   │   ├── 2012_Typhoon_Pablo-event_description.json
│   │   ├── 2012_Typhoon_Pablo-tweetids_entire_period.csv
│   │   ├── 2012_Typhoon_Pablo-tweets_labeled.csv
│   │   └── README.md
│   ├── 2012_Venezuela_refinery
│   │   ├── 2012_Venezuela_refinery-event_description.json
│   │   ├── 2012_Venezuela_refinery-tweetids_entire_period.csv
│   │   ├── 2012_Venezuela_refinery-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Alberta_floods
│   │   ├── 2013_Alberta_floods-event_description.json
│   │   ├── 2013_Alberta_floods-tweetids_entire_period.csv
│   │   ├── 2013_Alberta_floods-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Australia_bushfire
│   │   ├── 2013_Australia_bushfire-event_description.json
│   │   ├── 2013_Australia_bushfire-tweetids_entire_period.csv
│   │   ├── 2013_Australia_bushfire-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Bohol_earthquake
│   │   ├── 2013_Bohol_earthquake-event_description.json
│   │   ├── 2013_Bohol_earthquake-tweetids_entire_period.csv
│   │   ├── 2013_Bohol_earthquake-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Boston_bombings
│   │   ├── 2013_Boston_bombings-event_description.json
│   │   ├── 2013_Boston_bombings-tweetids_entire_period.csv
│   │   ├── 2013_Boston_bombings-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Brazil_nightclub_fire
│   │   ├── 2013_Brazil_nightclub_fire-event_description.json
│   │   ├── 2013_Brazil_nightclub_fire-tweetids_entire_period.csv
│   │   ├── 2013_Brazil_nightclub_fire-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Colorado_floods
│   │   ├── 2013_Colorado_floods-event_description.json
│   │   ├── 2013_Colorado_floods-tweetids_entire_period.csv
│   │   ├── 2013_Colorado_floods-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Glasgow_helicopter_crash
│   │   ├── 2013_Glasgow_helicopter_crash-event_description.json
│   │   ├── 2013_Glasgow_helicopter_crash-tweetids_entire_period.csv
│   │   ├── 2013_Glasgow_helicopter_crash-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_LA_airport_shootings
│   │   ├── 2013_LA_airport_shootings-event_description.json
│   │   ├── 2013_LA_airport_shootings-tweetids_entire_period.csv
│   │   ├── 2013_LA_airport_shootings-tweets_labeled.csv
│   │   ├── index-1.html
│   │   └── README.md
│   ├── 2013_Lac_Megantic_train_crash
│   │   ├── 2013_Lac_Megantic_train_crash-event_description.json
│   │   ├── 2013_Lac_Megantic_train_crash-tweetids_entire_period.csv
│   │   ├── 2013_Lac_Megantic_train_crash-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Manila_floods
│   │   ├── 2013_Manila_floods-event_description.json
│   │   ├── 2013_Manila_floods-tweetids_entire_period.csv
│   │   ├── 2013_Manila_floods-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_NY_train_crash
│   │   ├── 2013_NY_train_crash-event_description.json
│   │   ├── 2013_NY_train_crash-tweetids_entire_period.csv
│   │   ├── 2013_NY_train_crash-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Queensland_floods
│   │   ├── 2013_Queensland_floods-event_description.json
│   │   ├── 2013_Queensland_floods-tweetids_entire_period.csv
│   │   ├── 2013_Queensland_floods-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Russia_meteor
│   │   ├── 2013_Russia_meteor-event_description.json
│   │   ├── 2013_Russia_meteor-tweetids_entire_period.csv
│   │   ├── 2013_Russia_meteor-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Sardinia_floods
│   │   ├── 2013_Sardinia_floods-event_description.json
│   │   ├── 2013_Sardinia_floods-tweetids_entire_period.csv
│   │   ├── 2013_Sardinia_floods-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Savar_building_collapse
│   │   ├── 2013_Savar_building_collapse-event_description.json
│   │   ├── 2013_Savar_building_collapse-tweetids_entire_period.csv
│   │   ├── 2013_Savar_building_collapse-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Singapore_haze
│   │   ├── 2013_Singapore_haze-event_description.json
│   │   ├── 2013_Singapore_haze-tweetids_entire_period.csv
│   │   ├── 2013_Singapore_haze-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Spain_train_crash
│   │   ├── 2013_Spain_train_crash-event_description.json
│   │   ├── 2013_Spain_train_crash-tweetids_entire_period.csv
│   │   ├── 2013_Spain_train_crash-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_Typhoon_Yolanda
│   │   ├── 2013_Typhoon_Yolanda-event_description.json
│   │   ├── 2013_Typhoon_Yolanda-tweetids_entire_period.csv
│   │   ├── 2013_Typhoon_Yolanda-tweets_labeled.csv
│   │   └── README.md
│   ├── 2013_West_Texas_explosion
│   │   ├── 2013_West_Texas_explosion-event_description.json
│   │   ├── 2013_West_Texas_explosion-tweetids_entire_period.csv
│   │   ├── 2013_West_Texas_explosion-tweets_labeled.csv
│   │   └── README.md
│   └── README.md
├── CrisisLexT6
│   ├── 2012_Sandy_Hurricane
│   │   └── 2012_Sandy_Hurricane-ontopic_offtopic.csv
│   ├── 2013_Alberta_Floods
│   │   └── 2013_Alberta_Floods-ontopic_offtopic.csv
│   ├── 2013_Boston_Bombings
│   │   └── 2013_Boston_Bombings-ontopic_offtopic.csv
│   ├── 2013_Oklahoma_Tornado
│   │   └── 2013_Oklahoma_Tornado-ontopic_offtopic.csv
│   ├── 2013_Queensland_Floods
│   │   └── 2013_Queensland_Floods-ontopic_offtopic.csv
│   └── 2013_West_Texas_Explosion
│       └── 2013_West_Texas_Explosion-ontopic_offtopic.csv
├── CrisisMMD
│   ├── annotations
│   │   ├── california_wildfires_final_data.tsv
│   │   ├── hurricane_harvey_final_data.tsv
│   │   ├── hurricane_irma_final_data.tsv
│   │   ├── hurricane_maria_final_data.tsv
│   │   ├── iraq_iran_earthquake_final_data.tsv
│   │   ├── mexico_earthquake_final_data.tsv
│   │   └── srilanka_floods_final_data.tsv
│   └── Readme.txt
├── CrisisNLP_Crowdflower
│   ├── 2013_Pakistan_eq
│   │   ├── 2013_Pakistan_earthquake_CF_labeled_data.tsv
│   │   └── labeling-instructions.txt
│   ├── 2014_California_Earthquake
│   │   ├── 2014_California_Earthquake_CF_labeled_data.tsv
│   │   └── labeling-instructions.txt
│   ├── 2014_Chile_Earthquake_cl
│   │   ├── 2014_Chile_Earthquake_cl_labeled_data.tsv
│   │   └── labeling-instructions.txt
│   ├── 2014_Chile_Earthquake_en
│   │   ├── 2014_Chile_Earthquake_en_CF_labeled_data.tsv
│   │   └── labeling-instructions.txt
│   ├── 2014_ebola_cf
│   │   ├── 2014_ebola_CF_labeled_data.tsv
│   │   └── labeling-instructions.txt
│   ├── 2014_Hurricane_Odile_Mexico_en
│   │   ├── 2014_Odile_Hurricane_en_CF_labeled_data.tsv
│   │   └── labeling-instructions.txt
│   ├── 2014_India_floods
│   │   ├── 2014_India_floods_CF_labeled_data.tsv
│   │   └── labeling-instructions.txt
│   ├── 2014_Middle_East_Respiratory_Syndrome_en
│   │   ├── 2014_MERS_en_CF_labeled_data.tsv
│   │   └── labeling-instructions.txt
│   ├── 2014_Pakistan_floods
│   │   ├── 2014_Pakistan_floods_CF_labeled_data.tsv
│   │   └── labeling-instructions.txt
│   ├── 2014_Philippines_Typhoon_Hagupit_en
│   │   ├── 2014_Philippines_Typhoon_Hagupit_en_CF_labeled_data.tsv
│   │   └── labeling-instructions.txt
│   ├── 2015_Cyclone_Pam_en
│   │   ├── 2015_Cyclone_Pam_en_CF_labeled_data.tsv
│   │   └── labeling-instructions.txt
│   ├── 2015_Nepal_Earthquake_en
│   │   ├── 2015_Nepal_Earthquake_en_CF_labeled_data.tsv
│   │   └── labeling-instructions.txt
│   ├── README.txt
│   └── Terms of use.txt
├── CrisisNLP_Volunteers
│   ├── 2014_California_Earthquake
│   │   ├── 2014_California_Earthquake.csv
│   │   └── labeling-instructions.txt
│   ├── 2014_Chile_Earthquake_cl
│   │   ├── 2014_chile_earthquake_cl.csv
│   │   └── labeling-instructions.txt
│   ├── 2014_Chile_Earthquake_en
│   │   ├── 2014_Chile_Earthquake_en.csv
│   │   └── labeling-instructions.txt
│   ├── 2014_Hurricane_Odile_Mexico_en
│   │   ├── 2014_Odile_Hurricane_en.csv
│   │   └── labeling-instructions.txt
│   ├── 2014_Iceland_Volcano_en
│   │   ├── 2014_Iceland_Volcano_en.csv
│   │   └── labeling-instructions.txt
│   ├── 2014_Malaysia_Airline_MH370_en
│   │   ├── 2014_Malaysia_Airline_MH370_en.csv
│   │   └── labeling-instructions.txt
│   ├── 2014_Middle_East_Respiratory_Syndrome_en
│   │   ├── 2014_Middle_East_Respiratory_Syndrome_en.csv
│   │   └── labeling-instructions.txt
│   ├── 2014_Philippines_Typhoon_Hagupit_en
│   │   ├── 2014_Typhoon_Hagupit_en.csv
│   │   └── labeling-instructions.txt
│   ├── 2015_Cyclone_Pam_en
│   │   ├── 2015_Cyclone_Pam_en.csv
│   │   └── labeling-instructions.txt
│   ├── 2015_Nepal_Earthquake_en
│   │   ├── 2015_Nepal_Earthquake_en.csv
│   │   └── labeling-instructions.txt
│   ├── CrisisNLP_volunteers_labeled_data
│   ├── Landslides_Worldwide_en
│   │   ├── labeling-instructions.txt
│   │   └── Landslides_Worldwide_en.csv
│   ├── Landslides_Worldwide_esp
│   │   ├── labeling-instructions.txt
│   │   └── Landslides_Worldwide_esp.csv
│   └── Landslides_Worldwide_fr
│       ├── labeling-instructions.txt
│       └── LandSlides_Worldwide_fr.csv
├── ICWSM_2018
│   ├── nepal
│   │   ├── 2015_Nepal_Earthquake_dev.tsv
│   │   ├── 2015_Nepal_Earthquake_test.tsv
│   │   └── 2015_Nepal_Earthquake_train.tsv
│   └── queensland
│       ├── 2013_Queensland_Floods_dev.tsv
│       ├── 2013_Queensland_Floods_test.tsv
│       └── 2013_Queensland_Floods_train.tsv
├── ISCRAM_2013
│   └── Joplin_2011_labeled_data
│       ├── 01_personal-informative-other
│       │   └── a131709.csv
│       ├── 02_informative_caution-infosrc-donation-damage-other
│       │   └── a121571.csv
│       ├── README.txt
│       └── Terms of use.txt
└── SWDM_2013
    └── SWDM2013
        └── sandy2012_labeled_data
            ├── 01_personal-informative-other
            │   └── a143145.csv
            ├── 02_informative_caution-infosrc-donation-damage-other
            │   └── a144267.csv
            ├── README.txt
            └── Terms of use.txt
```

(Any files other that .csv and .tsv are not important)

Which resource from the earlier mentioned links correspond to which folder should be clear from the names of the downloaded files onces you download the resources. 

Note however, I **renamed** some of the csv/tsv files. I did this so that I could systematically organize according to disaster types (though I don't really make much use of the organization other than when doing some statistics. Ideally train-test split can be done on the basis on disaster type to test how well the model generalize to unseen disasters.). I **renamed** the files for the convinience of coding (I grouped tweets into disaster types from various sources based on their names, and I needed them all to follow a consistent naming pattern). In retrospect, I should have done everything through code instead of manually renaming the files - because now it becomes much harder for other people to set up the data.  


#### An example of how to set up the directory from the downloaded resources:

Let us say you downloaded "Labeled data of all the events annotated by volunteers" from [CrisisNLP Resource #1](https://crisisnlp.qcri.org/lrec2016/lrec2016.html)

The downloaded file is named: "CrisisNLP_volunteers_labeled_data.zip". 

From this you can figure out that the data corresponds to the data within "CrisisNLP_Volunteers" in the above directory tree. 
Next you can compare the original directory tree within "CrisisNLP_volunteers_labeled_data.zip" and the above directory tree under "CrisisNLP_Volunteers" and check out for differences in names. For example, you will find that in the original directory tree, there is a file called "2014_Hurricane_Odile_Mexico_en.csv" but in my directory tree the same file is "2014_Odile_Hurricane_en.csv". So you have to rename it. This has to be done for all the files, but on the plus side there should be only a few files to rename (in retrospect, should have kept a log to keep track of what needs to renamed). 

Also remove the csv/tsv files that do not appear in my directory tree. 

Releasing the full processed data would have made things a lot easier but Twitter privacy policies make that difficult. 


## Using your own data or your own data processing code

Instead of all the above hassle you can set up your own code for data extraction, aggregation, and processing. Ultimately, this may be the easiest thing to do for some of you given my clumsy setup. 

All you need to do is prepare three json files in this [folder](https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/tree/master/Classification/Processed_Data) - train_data.json, val_data.json, and test_data.json. Each file should have contents in the same format:

```
d = {}
d["tweet_ids"] = test_tweet_ids # list of tweet ids (optional; can be ignored - not used later on)
d["tweets"] = test_tweets # corresponding list of tweets
d["labels"] = test_labels # corresponding list of multi-class labels
d["binary_labels"] = test_binary_labels # corresponding list of binary class labels

with open("test_data.json", 'w') as outfile:
    json.dump(d, outfile)

```

In addition you would also need to prepare label_info.json

```
d = {}
d["labels2idx"] = labels2idx # label to integer dictionary eg. {"class 1": 0, "class 2": 1....}
d["binary_labels2idx"] = binary_labels2idx # binary label to integer dictionary eg. {"class 0": 0, "class 1": 1}
d["label_weights"] = label_weights # label -> class weight dictionary eg. {"class 1": 1.0, "class 2": 1.0,.....}
d["binary_label_weights"] = binary_label_weights  # binary label -> class weight dictionary eg. {"class 0": 1.0, "class 1": 1.0}

with open("label_info.json", 'w') as outfile:
    json.dump(d, outfile)
```


## Saving (Multilingual) BERT

Run [Save_pre_trained_locally.py](https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/blob/master/Classification/Save_pre_trained_locally.py) to download and locally save Multilingual BERT (cased, base) model through HuggingFace's library. Rest of the codes use the local directory to load BERT. So running this file is **necessary** before training or testing the models. 

You can also download a different model here, but if you use some other 'BERT-like' model (XLNet, RoBERTa, ELECTRA etc.) then you would need to change quite a few things. You can contact me for guidance. 


## Training



## Testing

## Import Errors



## Masked Language Modeling (MLM)

There is a [demo code](https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/blob/master/MLM/Demo.py) for masked language model training. One could build upon this code to set up MLM training with loads of unannotated Twitter data and build a 'Tweet-BERT'. BERT is mostly pre-trained on formal domains so pre-training on informal domains like Twitter can be potentially helpful - making BERT more 'familiar' with the distributional nature of informal text domains. It would be something interesting to try. 



(Under Construction)
