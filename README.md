## Introduction

This is a tool that generates summaries of conversational texts and identify action items from the summary.
<br>
This is an official entry to the [AWS Deep Learning Challenge 2022](https://amazon-ec2-dl1.devpost.com/).


## Installation

1. git clone https://github.com/ericasagun/AWS-DL-2022.git
2. Run
```bash
pip install -r requirements.txt
```


## Usage

1. Train models. Expected summarizer model size is ~2.5GB.
```bash
python src/action_item_classifier/train_classifier.py
python src/summarizer/train_summarizer.py
```
2. Verify model files are generated in models folder.
3. Run 
```bash
python src\main.py
```
4. Enter sample_convo.txt


## Dataset description

- [Samsum dataset](https://huggingface.co/datasets/samsum): The SAMSum dataset contains about 16k messenger-like conversations with summaries. Conversations were created and written down by linguists fluent in English. 
- [Action item dataset](https://github.com/vaibhavsanjaylalka/Action-Item-Detection/blob/master/train-data/sentence1.csv): Action items extracted from enron-email dataset


## Maintainers

- [Aldrin Lambon](https://github.com/aldrinlambon)
- [Erica Faye Sagun](https://github.com/ericasagun)
- [Myra Saet](https://github.com/myrasaet)
