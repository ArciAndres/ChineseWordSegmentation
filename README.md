# Chinese Word Segmentation -  Deep Learning solution

## Overview

Processing words and the connection between them is a key requirement for many tasks in language processing, and some languages, like Chinese, do not separate the words in written text. The objective of this work is to create a model that processes a text with no spaces, and generates the corresponding segmentation annotation, in BIES format, with deep learning tools, specially LSTM layers.

This repository provides an already trained model to perform the task on a given file, and the training codes if a new tuning or dataset wants to be used. 

**BIES format**: (**B**)eginning (**I**)nside (**E**)nd (**S**)ingle

**Output example:**

| Input                                                        | Output                                  | Result                                                       |
| ------------------------------------------------------------ | --------------------------------------- | ------------------------------------------------------------ |
| `theansweris42!`                                             | BIEBIIIIEBEBES                          | `the answer is 42 !`                                         |
| 分佈：<br/>主要分佈在熱帶及亞熱帶水域，<br/>溫帶水域亦有其活動記錄。 | BES<br/>BEBESBESBIEBES<br/>BEBESSSBEBES | 分佈 ：<br/>主要 分佈 在 熱帶 及 亞熱帶 水域 ，<br/>溫帶 水域 亦 有 其 活動 記錄 。 |

## Training



### Datasets 

Download the dataset: http://sighan.cs.uchicago.edu/bakeoff2005/
The full dataset contains four smaller datasets:

* AS (Traditional Chinese)
* CITYU (Traditional Chinese)
* MSR (Simplified Chinese)
* PKU (Simplified Chinese)

## Model



https://drive.google.com/drive/folders/1WBGOIS-VK7E8vpzOEUWD5vzXjQLjWM2g?usp=sharing