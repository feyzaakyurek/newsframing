# Multi-label and Multilingual News Framing Analysis

Code repository for the below paper.

Afra Feyza Akyürek, Lei Guo, Randa Elanwar, Margrit Betke, Prakash Ishwar and Derry T. Wijaya. Multi-label and Multilingual News Framing Analysis. In Proceedings of ACL 2020.

## Abstract

News framing refers to the strategy in which aspects of certain issues are highlighted in the news to promote a particular interpretation. In NLP, although recent works have studied framing in English news, few have studied how the analysis can  be extended to other languages and in a multi-label setting. In this work, we explore multilingual transfer learning to detect multiple frames from just the news headline in a genuinely low-resource setting where there are few/no frame annotations in the target language. We propose a novel method that can leverage very basic resources consisting of a dictionary and few annotations in a target language to detect frames in the language. Our method performs comparably or better than translating the entire target language headline to the source language for which we have annotated data. This opens up an exciting new capability of scaling up frame analysis to many languages, even those without existing translation technologies. Lastly, we apply our method to detect frames on the issue of U.S. gun violence in multiple languages and  obtain interesting insights on the relationship between different frames of the same issue across different countries with different languages.


## Reproducing Experiments

### Requirements

* python 3.7.3   
* pytorch 1.1
* cuda 10.1
* transformers 2.1.1


### Running Experiments

1. Go to `run_public.sh` and update `OUTPUT_GLOBAL_DIR`, `DATA_GLOBAL_DIR`, `CACHE_GLOBAL_DIR` and optionally `BASELINE_DATA_GLOBAL_DIR` if running Table 1 Experiments 3 and 4.
2. For training and evaluation run

```
sh run_public.sh EXP_NAME
```

replacing `EXP_NAME` with the name of the experiment you'd like to run. See the below table.

| Experiment           | Description                            | `EXP_NAME` |
|----------------------|----------------------------------------|-----------------|
| Table 1 Experiment 1 | Multiclass English BERT                | Table1Exp1      |
| Table 1 Experiment 2 | Multiclass Multi-BERT                  | Table1Exp2      |
| Table 1 Experiment 3 | Not available yet.                     |                 |
| Table 1 Experiment 4 | Not available yet.                     |                 |
| Table 1 Experiment 5 | Multi-label English BERT ML Focal Loss | Table1Exp5      |
| Table 1 Experiment 6 | Multi-label Multi-BERT ML Focal Loss   | Table1Exp6      |
| Table 1 Experiment 7 | Multi-label Multi-BERT BCE Loss        | Table1Exp7      |
| Table 2 Experiment 1 | Multi-label Multi-BERT Train EN Test DE| Table2Exp1DE    |
| | Multi-label Multi-BERT Train EN Test AR|Table2Exp1AR |
| | Multi-label Multi-BERT Train EN Test TR|Table2Exp1TR |
| Table 2 Experiment 2 | Multi-label Multi-BERT Code-Switched Train EN Test DE| Table2Exp2DE    |
| | Multi-label Multi-BERT Code-Switched Train EN Test AR|Table2Exp2AR |
| | Multi-label Multi-BERT Code-Switched Train EN Test TR|Table2Exp2TR |
|...|

Please see `run_public.sh` for a full list of experiments. The variable `EXP_NAME` is generally intuitive.

### Questions?

Please feel free to reach me at akyurek AT bu DOT edu
