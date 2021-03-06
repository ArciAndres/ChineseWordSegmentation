# Chinese Word Segmentation -  Deep Learning solution

## Overview

Processing words and the connection between them is a key requirement for many tasks in language processing, and some languages, like Chinese, do not separate the words in written text. The objective of this work is to create a model that processes a text with no spaces, and generates the corresponding segmentation annotation, in BIES format, with deep learning tools, specially LSTM layers.

This repository provides an already trained model to perform the task on a given file, and the training codes if a new tuning or dataset wants to be used. 

**BIES format**: (**B**)eginning (**I**)nside (**E**)nd (**S**)ingle

**Output example:**

| Language            | Input                                                        | Output                                  | Result                                                       |
| ------------------- | ------------------------------------------------------------ | --------------------------------------- | ------------------------------------------------------------ |
| English (example)   | `theansweris42!`                                             | BIEBIIIIEBEBES                          | `the answer is 42 !`                                         |
| Chinese (this repo) | 分佈：<br/>主要分佈在熱帶及亞熱帶水域，<br/>溫帶水域亦有其活動記錄。 | BES<br/>BEBESBESBIEBES<br/>BEBESSSBEBES | 分佈 ：<br/>主要 分佈 在 熱帶 及 亞熱帶 水域 ，<br/>溫帶 水域 亦 有 其 活動 記錄 。 |

## Training

Much of the preprocessing and training was performed in [Google Colab](https://colab.research.google.com/), with notebooks that explain step-by-step the whole process. Feel free to explore them in `resources/FilesUsedForTraining`.

### Datasets

Download the dataset: http://sighan.cs.uchicago.edu/bakeoff2005/
The full dataset contains four smaller datasets:

* AS (Traditional Chinese)
* CITYU (Traditional Chinese)
* MSR (Simplified Chinese)
* PKU (Simplified Chinese)

### Model

Based on paper of reference model: https://aclweb.org/anthology/D18-1529

(Ma et al., 2018) State-of-the-art Chinese Word Segmentation with Bi-LSTMs:

| Model                                                        |
| ------------------------------------------------------------ |
| <img src="media/model.svg" alt="model" width=800px />        |
| Figure: Bi-LSTM models: (a) non-stacking, (b) stacking. Blue circles are input (char and char bigram) embeddings. <br />Red squares are LSTM cells. BIES is a 4-way softmax. |

**Training**

Training was performed with 30% of the merged dataset, and final model with 100% after tuning parameters. The following figures depict one of the performed grid search variations. Static learning rate of 0.0005, and variable dropout in recurrent units with values `[0, 0.1,0.4,0.6]`. Shows sings of overfitting. Read the complete information report in `NLP_HW1_Report.pdf`.

![image-20210110164259523](media/image-trainings.png)

## Testing

**Pretrained model**


A pretrained model and vocabulary set is provided in this folder:

https://drive.google.com/drive/folders/1WBGOIS-VK7E8vpzOEUWD5vzXjQLjWM2g?usp=sharing

The files should be located in `resources/Model` to be used by the `predict.py` and `score.py` scripts.

### Predict

To process an file through the segmenting model:

```
cd ChineseWordSegmentation
python code/predict.py input_path ________ output_path __________ resources_path resources
```

The `output_file` path should have the desired separation indication. 

### Scoring

If a `gold_file` is available, the accuracy of the prediction can be tested on the `score` function. It returns the precision of the model's predictions w.r.t. the gold standard (i.e. the tags of the correct word segmentation).

```
Example:
    predictions_iter = ["BEBESBIIE", "BIIIEBEBESS"]
    gold_iter = ["BEBIEBIES", "BIIESBEBESS"]
    output: 0.7
```

Usage

````
python code/score.py prediction_file ___________ gold_file ______________
````

