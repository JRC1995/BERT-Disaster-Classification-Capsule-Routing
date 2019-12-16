# Disaster-related Tweet Classification with BERT
#### Exploring Attention-based and Capsule-Routing-based Layer Aggregation and Hidden state Aggregation

See Project Report [here](https://drive.google.com/file/d/101NgTXm7zicqhrxr9e6n3bWPBLzFK50o/view?usp=sharing) and Presentation
Slides [here](https://docs.google.com/presentation/d/1R8PV1bEQqgApC2vYTMQhP0pXRvDjZ5qXnsOaLhyB31w/edit?usp=sharing)

## Main Requirements

* Numpy
* Pytorch 1.3
* HuggingFace's [Transformers](https://github.com/huggingface/transformers) v2.2.2

## Credits

Besides HuggingFace's library we used [Heinsen Routing](https://github.com/glassroom/heinsen_routing) as a reference.

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

We prepare annotations for both multi-class classification and binary classification. We train for both in a multi-task framework.

## Label Reduction Scheme

See this [code](https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/blob/master/Classification/Process_Data/Process_3rd_Stage.py) for the exact label reduction scheme. 


## Data Processing

First, the data can be either downloaded from the above links or collected from the tweet ids I make publicly available:
https://github.com/JRC1995/BERT-Disaster-Classification-Capsule-Routing/tree/master/Classification/Processed_Data

However, some of the tweet ids are not exactly tweet ids but some other ids (especially for the data from Resource #2 and Resource #3). You can get the same data by downloading from resource #2 and # 3 or you can also just filter them out (check for the length of tweet ids - their length will be much less) and ignore them. Should not make a lot of difference.

There are also some issues with extracting data in our codes from resource #2 and #3. Multiple rows get concatenated into one string. I noticed it too late, and although I could try to fix it, I didn't because one 1 or 2 samples were like this. 

Now if you download form tweet ids, they are already annotated in the json files. If you load the JSON file in some variable "json_data" then you can access the list of tweet ids from:
```
json_data["tweet_ids"]
```
You can access the corresponding multi-class labels from:
```
json_data["labels"]
```
You can access the corresponding binary labels from:
```
d["binary_labels"]
```




(Under Construction)
