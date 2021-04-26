# Task 2: Detecting Previously Fact-Checked Claims

This repository contains the _dataset_, _format checker, scorer and baselines_ for the [CLEF2021-CheckThat! Task 2](https://sites.google.com/view/clef2021-checkthat/tasks/task-2-claim-retrieval). 
Given a check-worthy claim in the form of a tweet, and a set of previously fact-checked claims, rank the previously fact-checked claims in order of usefulness to fact-check the input claim.


````
TBA
````

This file contains the basic information regarding the CLEF2021-CheckThat! Task 2
on check-worthiness on tweets provided for the CLEF2021-CheckThat! Lab
on "Automatic Detecting Check-Worthy Claims, Previously Fact-Checked Claims, and Fake News".
The current version are listed bellow corresponds to the release of the training and dev data sets.

__Table of contents:__
- [Evaluation Results](#evaluation-results)
- [List of Versions](#list-of-versions)
- [Contents of the Task 2 Directory](#Contents-of-the-Task-2-Directory)
- [Input Data Format](#input-data-format)
	- [Verified Claims (**Subtask-2A** and **Subtask-2B**)](#verified-claims-subtask-2a-and-subtask-2b)
		- [Verified Claims (Subtask-2A) - Arabic](#verified-claims-subtask-2a-arabic)
		- [Verified Claims (Subtask-2A) - English](#verified-claims-subtask-2a-english)
		- [Verified Claims (Subtask-2B) - English](#verified-claims-subtask-2b-english)
	- [Queries File (**Subtask-2A** and **Subtask-2B**)](#queries-file-subtask-2a-and-subtask-2b)
	- [Qrels File (**Subtask-2A** and **Subtask-2B**)](#qrels-file-subtask-2a-and-subtask-2b)
	- [Tweet Objects (**Subtask-2A** Only)](#tweet-objects-subtask-2a-only)
	- [Deabte/Speeches Transcripts (**Subtask-2B** Only)](#debatespeeches-transcripts-subtask-2b-only)
- [Output Data Format](#output-data-format)
- [Format Checkers](#format-checkers)
- [Scorers](#scorers)
- [Evaluation Metrics](#evaluation-metrics)
- [Baselines](#baselines)
	- [Random Baseline](#random-baseline)
	- [Elasticsearch Baseline](#elasticsearch-baseline)
- [Credits](#credits)

# Evaluation Results

TBA

# List of Versions

* __subtask-2b--english-v1.0 [2021/02/22]__ - Training/Dev data for subtask-2a--english released. Containing 562 iclaim--vclaim pair for train and 140 iclaim--vclaim pair for dev.


# Contents of the Task 2 Directory
We provide the following files:

* Main folder: [data](./data)
  * Subfolder: [subtask-2A--english](./data/subtask-2a--english)
  * Subfolder: [subtask-2A--arabic](./data/subtask-2a--arabic)
  * Subfolder: [subtask-2B--english](./data/subtask-2b--english)
* Main folder: [baselines](./baselines)<br/>
	Contains scripts provided for baseline models of the tasks
* Main folder: [formet_checker](./format_checker)<br/>
	Contains scripts provided to check format of the submission file
* Main folder: [scorer](./scorer)<br/>
	Contains scripts provided to score output of the model when provided with label (i.e., dev set).
* [README.md](TASK_README.md) <br/>
	This file!


# Input Data Format

The format used in the task is inspired from [Text REtrieval Conference (TREC)](https://trec.nist.gov/)'s campaigns for information retrieval (a description of the TREC format can be found [here](https://github.com/joaopalotti/trectools#file-formats)).

Both subtasks share the same file formats with minor differences. There are ## file types:
- [Verified Claims (**Subtask-2A** and **Subtask-2B**)](#verified-claims-subtask-2a-and-subtask-2b)
- [Queries File (**Subtask-2A** and **Subtask-2B**)](#queries-file-subtask-2a-and-subtask-2b)
- [Qrels File (**Subtask-2A** and **Subtask-2B**)](#qrels-file-subtask-2a-and-subtask-2b)
- [Tweet Objects (**Subtask-2A** Only)](#tweet-objects-subtask-2a-only)
- [Deabte/Speeches Transcripts (**Subtask-2B** Only)](#debatespeeches-transcripts-subtask-2b-only)

## Verified Claims (Subtask-2A and Subtask-2B)

To verify the input claims from tweets or debates we use a set of previously fact-checked claims (we call vclaim). 
For each subtask and all languages the subtask is offered in we provide a set of previously fact-checked vclaims. 
Each vclaim is represented in one json file. 

Different subtasks and language variation contain different fields. 

### Verified Claims (Subtask-2A) - Arabic 

TAB separated file where each row in the file has the following format

> vclaim_id <TAB> vclaim <TAB> title

where: <br>

* vclaim_id: unique ID of the verified claim <br/>
* vclaim: text of the verified claim <br/>
* title: title of the article providing justification for the vclaim truth_label <br/>

### Queries File (Subtask-2A) - Arabic 

TAB separated file where each row in the file has the following format

> tweet_id <TAB> tweet_text

where: <br>

* tweet_id: unique ID for a given tweet <br/>
* tweet_text: text of the input claim (tweet) <br/>

### Verified Claims (Subtask-2A) - English
For each vclaim we provide the following, 

```
{
	vclaim_id 	[unique ID of the verified claim], 
	vclaim 		[text of the verified claim], 
	title 		[title of the article providing justification for the vclaim truth_label]
}
```

Example:
> TBA <br/>

### Verified Claims (Subtask-2B) - English
For each vclaim we provide the following entries, 

```
{
	vclaim_id 	[unique ID of the verified claim], 
	vclaim 		[text of the verified claim], 
	date 		[the date the verified claim was verified],
	truth_label	[the truth verdict given by the journalists], 
	speaker 	[The original speaker or source for the vclaim], 
	url 		[The link to the article providing justification for the vclaim truth_label], 
	title 		[title of the article providing justification for the vclaim truth_label], 
	text 		[text of the article providing justification for the vclaim truth_label]
}
```

Example:
```
{
	"vclaim_id": 0
	"vclaim": "Schuyler VanValkenburg co-sponsored a bill that would have \"allowed abortion until the moment of birth.\"",
	"url": "/factchecks/2019/oct/30/gaydonna-vandergriff/vandergriff-inaccurately-describes-bill-would-have/",
	"speaker": "GayDonna Vandergriff",
	"truth_label": false,
	"date": "stated on October 22, 2019 in a campaign mailer.",
	"title": "Vandergriff inaccurately describes bill that would have eased late-term abortion laws",
	"text": "Republican GayDonna Vandergriff has turned to abortion in her effort to portray ..."
}
```

## Queries File (Subtask-2A and Subtask-2B)

TAB separated file with the input claims that we refer to as `iclaim`. 
A row of the file has the following format

> iclaim_id <TAB> iclaim

where: <br>

* iclaim_id: unique ID for a given iclaim <br/>
* iclaim: text of the input claim (tweet or a sentence from a debate/speech) <br/>

Example: <br/>
| iclaim_id | iclaim |
| --- | --- |
| tweet-en-0008 | im screaming. google featured a hoax article that claims Minecraft is being shut down in 2020 pic.twitter.com/ECRqyfc8mI — Makena Kelly (@kellymakena) January 2, 2020 |
| tweet-en-0335  | BREAKING: Footage in Honduras giving cash 2 women & children 2 join the caravan & storm the US border @ election time. Soros? US-backed NGOs? Time to investigate the source! pic.twitter.com/5pEByiGkkN — Rep. Matt Gaetz (@RepMattGaetz) October 17, 2018 |
| tweet-en-0622 | y’all really joked around so much that tide put their tide pods in plastic boxes…smh pic.twitter.com/Z44efALcX5 — ㅤnavid (@NavidHasan\_) January 13, 2018 |
|...

**NOTE:** You can identify which debate used by looking at the `iclaim_id`. 
The `iclaim_id` will be in the format of `<debate_file_name>_<line_number_from_debate>`. 
For example the input claim said in the third presidential 2012 debate (20121023_third_presidential_debate.tsv) mentioned at line 6 will have a `vclaim_id=20121023_third_presidential_debate_006`. 


## Qrels File (Subtask-2A and Subtask-2B)


A TAB-separated file containing all the pairs of input claim and verified claim such that the verified claim (__vclaim_id__) proves the input claim (__iclaim_id__).

> iclaim_id <TAB> 0 <TAB> vclaim_id <TAB> relevance

where: <br/>

* iclaim_id: unique ID for a given iclaim. iclaim details found in the queries file. <br/>
* 0: literally 0 (this column is needed to comply with the TREC format).
* vclaim_id: unique ID for a given verified claim. Details on the verified claim are in qrels file <br/>
* relevance: 1 if the verified claim whose id is __vclaim_id__ proves the input claim with id __iclaim_id__; 0 otherwise.

__Note__: In the qrels file only pairs with relevance = 1 are reported. Relevance = 0 is assumed for all pairs not appearing in the qrels file.

Example:
| iclaim_id | 0 | vclaim_id | relevance |
|---|---|---|---|
| tweet-en-0422 | 0 | vclaim-sno-00092 | 1 |
| tweet-en-0538 | 0 | vclaim-sno-00454 | 1 |
| tweet-en-0221 | 0 | vclaim-sno-00012 | 1 |
| tweet-en-0137 | 0 | vclaim-sno-00504 | 1 |
|... |

## Tweet Objects (Subtask-2A Only)

Specific for subtask-2a we provide the tweet object obtained using the tweeter API. 

__Note__: Not all the input tweets have tweeter API object as some tweets were deleted after the time of crawling. 


## Debate/Speeches Transcripts (Subtask-2B Only)
We also provide the full transcripts of the debates/speeches from which the iclaim were obtained from. 

Similar to the debate format of (subtask-1B)[../task1#subtask-1b-check-worthiness-of-debatesspeeches]. The debate files are TAB-separated TSV files with four fields:

> line_number <TAB> speaker <TAB> text

Where: <br>
* line_number: the line number (starting from 1) <br/>
* speaker: the person speaking (a candidate, the moderator, or "SYSTEM"; the latter is used for the audience reaction) <br/>
* text: a sentence that the speaker said <br/>

The text encoding is UTF-8.

Example:

>  ... <br/>
>  65  TRUMP So we're losing our good jobs, so many of them.<br/>
>  66  TRUMP When you look at what's happening in Mexico, a friend of mine who builds plants said it's the eighth wonder of the world. <br/>
>  67  TRUMP They're building some of the biggest plants anywhere in the world, some of the most sophisticated, some of the best plants.<br/>
>  ...

# Output Data Format

Each row of the result file is related to a pair _iclaim_ and _vclaim_ and intuitively indicates the ranking of the verified claim with respect to the input claim.
Each row has the following format:
>iclaim_id <TAB> 0 <TAB> vclaim_id <TAB> rank <TAB> score <TAB> tag

where <br>

* iclaim_id: ID of the iclaim
* Q0: random value (it is needed to comply with the TREC format).
* vclaim_id: ID of the verified claim found in the verified claims file (data/verified_claims.qrels.tsv)
* rank: the rank of the pair given based on the scores of all possible pairs for a given _tweet_id_. (Not taken into account when calculating metrics. Always equal to 1)
* score: the score given by your model for the pair _tweet_id_ and _vclaim_id_
* tag: string identifier of the team.

 For example:
|tweet_id  |  Q0  |  vclaim_id  |  rank  |  score  | tag |
| --- | --- | --- | --- | --- | --- |
| tweet-en-0359 | Q0 | vclaim-sno-00303  | 1 | 1.1086285 | elastic |
| tweet-en-0476 | Q0 | vclaim-sno-00292  | 1 |  4.680018 | elastic
| tweet-en-035  | Q0 | vclaim-sno-00373  | 1 |  5.631936 | elastic
| tweet-en-0474 | Q0 | vclaim-sno-00352  | 1 |  0.883034 | elastic
| tweet-en-0174 | Q0 | vclaim-sno-00408  | 1 |  0.980456 | elastic
| ...

Your result file **MUST** have at most 1 unique pair of tweet_id and vclaim_id. You can skip pairs if you deem them not relevant.

## Example of Output Ranking

The following is an example ranking of verified claims for given English tweet 

Let's take random tweet from the data set:
> __iclaim_id__:  tweet-en-0251 <br>
> __iclaim__: A big scandal at @ABC News. They got caught using really gruesome FAKE footage of the Turks bombing in Syria. A real disgrace. Tomorrow they will ask softball questions to Sleepy Joe Biden’s son, Hunter, like why did Ukraine & China pay you millions when you knew nothing? Payoff? — Donald J. Trump (@realDonaldTrump) October 15, 2019

Using the content of the tweet, your model should give the highest score to the verified claim, that matches the tweet in the [Qrels file](#qrels-file). In this case the verified claim is:

> __vclaim_id__: vclaim-sno-00115 <br>
> __vclaim__: ABC News mistakenly aired a video from a Kentucky gun range during its coverage of Turkey's attack on northern Syria in October 2019.

Example of top 5 ranked verfied claims from the baseline model in this repository:
| vclaim | score |
| ---    | --- |
| ABC News mistakenly aired a video from a Kentucky gun range during its coverage of Turkey's attack on northern Syria in October 2019. | 21.218384 |
| In a speech to U.S. military personnel, President Trump said if soldiers were real patriots, they wouldn't take a pay raise. | 19.962847 |
| Former President Barack Obama tweeted: "Ask Ukraine if they found my birth certificate." | 19.414398 |
| Mark Twain said, "Do not fear the enemy, for your enemy can only take your life. It is far better that you fear the media, for they will steal your HONOR." | 16.810490 |
| Dolly Parton wrote "Jolene" and "I Will Always Love You" in one day.  | 16.005116 |

**NOTE:** All subtasks and language variations of this task uses the same output format. 

# Format Checkers
The checker for the subtask is located in the [format_checker](./format_checker) module of the project.
To launch the baseline script you need to install packages dependencies found in [requirement.txt](./requirement.txt) using the following:
> pip3 install -r requirements.txt <br/>

The format checker verifies that your generated results files complies with the expected format.
To launch it run: 
> python format_checker/main.py --pred-files-path=<path_to_result_file_1 path_to_result_file_2 ... path_to_result_file_n> <br/>

`--pred-files-path` take a single string that contains a space separated list of file paths. The lists may be of arbitrary positive length (so even a single file path is OK) but their lengths must match.

__<path_to_result_file_n>__ is the path to the corresponding file with participants' predictions, which must follow the format, described in the [Output Data Format](#output-data-format) section.

Note that the checker can not verify whether the prediction files you submit contain all lines / claims), because it does not have access to the corresponding gold file.

# Scorers

The scorer for the subtask is located in the [scorer](./scorer) module of the project.
To launch the script you need to install packages dependencies found in [requirement.txt](./requirement.txt) using the following:
> pip3 install -r requirements.txt <br/>

Launch the scorer for the subtask as follows:
> python3 scorer/main.py --gold-file-path=<path_gold_file> --pred-file-path=<predictions_file> <br/>

The scorer invokes the format checker for the subtask to verify the output is properly shaped.
It also handles checking if the provided predictions file contains all lines/tweets from the gold one.


# Evaluation Metrics
This task is evaluated as a ranking task. Ranked list per claim will be evaluated using ranking evaluation measures (MAP@k for k=1,3,5,10,20,all, MRR and Recall@k for k=1,3,5,10,20). Official measure is MAP@5.


# Baselines
The [baselines](./baselines) module contains a random and a simple BM25 IR baseline for the task.

| Model | subtask-2A--English | subtask-2A--Arabic | subtask-2B--English |
| :---: | :---: | :---: | :---: |
| Random Baseline |  |  | 0.0000 |
| Ngram Baseline  |  |  | 0.3207 |



To launch the baseline script you need to install packages dependencies found in [requirement.txt](./requirement.txt) using the following:
> pip3 install -r requirements.txt <br/>

## Random Baseline
To launch the random baseline script run the following:
> python3 baselines/random_baseline.py --train-file-path=<path_to_your_training_data> --test-file-path=<path_to_your_test_data_to_be_evaluated> --lang=<language_of_the_task_2> --subtask=<subtask_you_are_working_on> --vclaims-dir-path=<path_to_your_directory_of_vclaims> <br/>


## Elasticsearch Baseline 
To use the Elastic Search baseline you need to have a locally running Elastic Search instance.
You can follow [this](https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started-install.html) article for Elastic Search installation. You can then run elasticsearch using the following command:
> /path/to/elasticsearch

To launch the elasticsearch baseline script run the following:
> python3 baselines/elasticsearch_baseline.py --train-file-path=<path_to_your_training_data> --test-file-path=<path_to_your_test_data_to_be_evaluated> --lang=<language_of_the_task_2> --subtask=<subtask_you_are_working_on> --vclaims-dir-path=<path_to_your_directory_of_vclaims> --iclaims-file-path=<path_to_queries_iclaim_file> <br/>

**NOTE:** There are other parameters you can play with in the elasticsearch baseline script. 


# Credits

Task 2 Organizers: TBA

Task website: https://sites.google.com/view/clef2021-checkthat/tasks/task-2-claim-retrieval

Contact:   clef-factcheck@googlegroups.com
