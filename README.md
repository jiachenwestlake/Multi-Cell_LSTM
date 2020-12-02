# Multi-Cell_LSTM
Multi-Cell Compositional LSTM for NER Domain Adaptation, code for ACL 2020 paper.

## Introduction
Cross-domain NER is a challenging yet practical problem. Entity mentions can be highly different across domains. However, the correlations between entity types can be relatively more stable across domains. We investigate a multi-cell compositional LSTM structure for multi-task learning, modeling each entity type using a separate cell state. With the help of entity typed units, cross-domain knowledge transfer can be made in an entity type level. Theoretically, the resulting distinct feature distributions for each entity type make it more powerful for cross-domain transfer. Empirically, experiments on four few-shot and zero-shot datasets show our method signiﬁcantly outperforms a series of multi-task learning methods and achieves the best results.
<br> <br>
For more details, please refer to our paper:
<br><br>
[Multi-Cell Compositional LSTM for NER Domain Adaptation](https://www.aclweb.org/anthology/2020.acl-main.524.pdf)

## Overall Structure
![Overall Structure](https://github.com/jiachenwestlake/Multi-Cell_LSTM/blob/master/overall_structure.PNG)

**The Entity Typed cells (ET cells) correspond to the source- and target-domain entity types (_including O, which is used as the outside tagger in NER_).**

## Requirements
```
Python 3 
PyTorch 1.0+
allennlp 0.8.2 (Optional)
pytorch-pretrained-bert 0.6.1 (Optional)
```

## Word Embeddings and Pretrained LM Weights
GloVe 100-dimension word vectors (Download from [*Here*](https://pan.baidu.com/s/1kgwVCGkbmcUVYyLb8VlXeQ) with key `ifyk`) <br>
PubMed 200-dimension word vectors (Refer to [*Here*](https://www.aclweb.org/anthology/W16-2922/)) <br>
ELMo Weights (Download from [*Here*](https://pan.baidu.com/s/1I4589wAsI-pivRcA4kTavg) with key `a9h6`) <br>
BERT-base Weights (Download from [*Here*](https://pan.baidu.com/s/1c1sK_J-cbJvtz52WyA9IBg) with key `gbn1`) <br>
BioBERT-base Weights (Download from [*Here*](https://pan.baidu.com/s/1Mj81A9SRtXwQ_SZoV8vHfA) with key `zsep`)

## DataSet
### Supervised Domain Adaptation (SDA):
CoNLL-2003 English NER data (In: `SDA/data/conll03_En`) <br>
Broad Twitter corpus (In: `SDA/data/broad_twitter_corpus`) (or download from [*Here*](https://pan.baidu.com/s/1wbbZB7MHhTcsWe2UilT28Q) with key `0yko`)<br>
[BioNLP'13PC and BioNLP'13CG](https://github.com/cambridgeltl/MTL-Bioinformatics-2016/tree/master/data) dataset <br>
Twitter corpus (Refer to [*Here*](https://www.aclweb.org/anthology/P18-1185.pdf)) (Download from [*Here*](https://pan.baidu.com/s/1g06HK3PHbb-iMFw1BiTiBQ) with key `bn75`)

### Unsupervised Domain Adaptation (UDA):
CoNLL-2003 English NER data (In: `SDA/data/conll03_En`). <br>
CBS SciTech News (test set) (In: `UDA/data/tech/tech.test`). <br>

## LM raw data
SciTech news domain raw data [Download](https://pan.baidu.com/s/1_gv5eop_lL3HEUbhj8JqQg) with key `4834`, and put it in `UDA/data/tech`. 

## Entity dictionary used in UDA
The named entity dictionary is collected by [Peng et. al.](https://arxiv.org/pdf/1906.01378.pdf) and In `UDA/data/tech/conll2003_dict`.

## Usage
### Train
`SDA` and `UDA` can use the following command to make it run. <br>
```
python main.py --config train.config
```
The file `train.config` contains dataset path and model hyperparameters.

### Decode
`SDA` and `UDA` can use the following command to make it run. <br>
```
python main.py --config decode.config
```
The file `decode.config` contains dataset path and paths for tarined models. <br>
For example, you can download our trained [models](https://pan.baidu.com/s/1Tdoli7QiPl2SUL6N6w4X3g) with key `matp`, unzip the two files `.dset` and `.model` and put them into `SDA/saved_models`. Then you can use the above comment to get our reported result on the broad twitter corpus. UDA [models](https://pan.baidu.com/s/1ji7REqVJOdrAfI4v5Mv3Mw) with key `2s6n` are decoded similarly.


## Cite:
If you use our code, please cite our paper as follows:
```
@inproceedings{jia-zhang-2020-multi,
    title = "Multi-Cell Compositional {LSTM} for {NER} Domain Adaptation",
    author = "Jia, Chen  and  Zhang, Yue",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.524",
    pages = "5906--5917"
}
```
