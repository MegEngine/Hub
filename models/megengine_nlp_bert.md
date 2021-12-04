---
template: hub1
title: BERT for Finetune
summary:
    en_US: Bidirectional Encoder Representation from Transformers (BERT)
    zh_CN: BERT
author: MegEngine Team
tags: [nlp]
github-link: https://github.com/MegEngine/Models/tree/master/official/nlp/bert
---

```python
import megengine.hub as hub
model = megengine.hub.load("megengine/models", "chinese_L_12_H_768_A_12", pretrained=True)
# or any of these variants
# model = megengine.hub.load("megengine/models", "cased_L_12_H_768_A_12", pretrained=True)
# model = megengine.hub.load("megengine/models", "cased_L_24_H_1024_A_16", pretrained=True)
# model = megengine.hub.load("megengine/models", "chinese_L_12_H_768_A_12", pretrained=True)
# model = megengine.hub.load("megengine/models", "multi_cased_L_12_H_768_A_12", pretrained=True)
# model = megengine.hub.load("megengine/models", "uncased_L_12_H_768_A_12", pretrained=True)
# model = megengine.hub.load("megengine/models", "uncased_L_24_H_1024_A_16", pretrained=True)
# model = megengine.hub.load("megengine/models", "wwm_cased_L_24_H_1024_A_16", pretrained=True)
# model = megengine.hub.load("megengine/models", "wwm_uncased_L_24_H_1024_A_16", pretrained=True)
```

<!-- section: zh_CN -->

这个项目中, 我们用MegEngine重新实现了Google开源的BERT模型.

我们提供了以下预训练模型供用户在不同的下游任务中进行finetune.

* `wwm_cased_L-24_H-1024_A-16`
* `wwm_uncased_L-24_H-1024_A-16`
* `cased_L-12_H-768_A-12`
* `cased_L-24_H-1024_A-16`
* `uncased_L-12_H-768_A-12`
* `uncased_L-24_H-1024_A-16`
* `chinese_L-12_H-768_A-12`
* `multi_cased_L-12_H-768_A-12`

模型的权重来自Google的pre-trained models, 其含义也与其一致, 用户可以直接使用`megengine.hub`轻松的调用预训练的bert模型, 以及下载对应的`vocab.txt`与`bert_config.json`. 我们在[models](https://github.com/megengine/models/official/nlp/bert)中还提供了更加方便的脚本, 可以通过任务名直接获取到对应字典, 配置, 与预训练模型.

```python
import megengine.hub as hub
import urllib
import urllib.request
import os

DATA_URL = 'https://data.megengine.org.cn/models/weights/bert'
CONFIG_NAME = 'bert_config.json'
VOCAB_NAME = 'vocab.txt'
MODEL_NAME = {
    'wwm_cased_L-24_H-1024_A-16': 'wwm_cased_L_24_H_1024_A_16',
    'wwm_uncased_L-24_H-1024_A-16': 'wwm_uncased_L_24_H_1024_A_16',
    'cased_L-12_H-768_A-12': 'cased_L_12_H_768_A_12',
    'cased_L-24_H-1024_A-16': 'cased_L_24_H_1024_A_16',
    'uncased_L-12_H-768_A-12': 'uncased_L_12_H_768_A_12',
    'uncased_L-24_H-1024_A-16': 'uncased_L_24_H_1024_A_16',
    'chinese_L-12_H-768_A-12': 'chinese_L_12_H_768_A_12',
    'multi_cased_L-12_H-768_A-12': 'multi_cased_L_12_H_768_A_12'
}

def download_file(url, filename):
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

def create_hub_bert(model_name, pretrained):
    assert model_name in MODEL_NAME, '{} not in the valid models {}'.format(model_name, MODEL_NAME)
    data_dir = './{}'.format(model_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    vocab_url = '{}/{}/{}'.format(DATA_URL, model_name, VOCAB_NAME)
    config_url = '{}/{}/{}'.format(DATA_URL, model_name, CONFIG_NAME)

    vocab_file = './{}/{}'.format(model_name, VOCAB_NAME)
    config_file = './{}/{}'.format(model_name, CONFIG_NAME)

    download_file(vocab_url, vocab_file)
    download_file(config_url, config_file)

    config = BertConfig(config_file)

    model = hub.load(
        "megengine/models",
        MODEL_NAME[model_name],
        pretrained=pretrained,
    )

    return model, config, vocab_file
```

为了用户可以更加方便的使用预训练模型, 我们仅保留了模型的`BertModel`的部分, 在实际使用中, 可以将带有预训练的权重的`bert`模型作为其他模型的一部分, 在初始化函数中传入.

```python
class BertForSequenceClassification(Module):
    def __init__(self, config, num_labels, bert):
        self.bert = bert
        self.num_labels = num_labels
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids,
            attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss = cross_entropy_with_softmax(
                logits.reshape(-1, self.num_labels),
                labels.reshape(-1))
            return logits, loss
        else:
            return logits, None

bert, config, vocab_file = create_hub_bert('uncased_L-12_H-768_A-12', pretrained=True)
model = BertForSequenceClassification(config, num_labels=2, bert=bert)
```

所有预训练模型希望数据被正确预处理, 其要求与Google中的开源bert一致, 详细可以参考 [bert](https://github.com/google-research/bert), 或者参考在[models](https://github.com/megengine/models/official/nlp/bert)中提供的样例.

### 模型描述

我们在[models](https://github.com/megengine/models/official/nlp/bert)中提供了简单的示例代码.
此示例代码在Microsoft Research Paraphrase（MRPC）数据集上对预训练的`uncased_L-12_H-768_A-12`模型进行微调.

我们的样例代码中使用了原始的超参进行微调, 在测试集中可以得到84％到88％的正确率.

### 参考文献

 - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova;


<!-- section: en_US -->
This repository contains reimplemented Google's BERT by MegEngine.

We provide the following pre-trained models for users to finetune in different tasks.

* `wwm_cased_L-24_H-1024_A-16`
* `wwm_uncased_L-24_H-1024_A-16`
* `cased_L-12_H-768_A-12`
* `cased_L-24_H-1024_A-16`
* `uncased_L-12_H-768_A-12`
* `uncased_L-24_H-1024_A-16`
* `chinese_L-12_H-768_A-12`
* `multi_cased_L-12_H-768_A-12`

The weight of the model comes from Google's pre-trained models, and its meaning is also consistent with it. Users can use `megengine.hub` to easily use the pre-trained bert model, and download the corresponding` vocab.txt` and `bert_config.json`. We also provide a convenient script in [models](https://github.com/megengine/models/official/nlp/bert), which can directly obtain the corresponding dictionary, configuration, and pre-trained model by task name. .

```python
import megengine.hub as hub
import urllib
import urllib.request
import os

DATA_URL = 'https://data.megengine.org.cn/models/weights/bert'
CONFIG_NAME = 'bert_config.json'
VOCAB_NAME = 'vocab.txt'
MODEL_NAME = {
    'wwm_cased_L-24_H-1024_A-16': 'wwm_cased_L_24_H_1024_A_16',
    'wwm_uncased_L-24_H-1024_A-16': 'wwm_uncased_L_24_H_1024_A_16',
    'cased_L-12_H-768_A-12': 'cased_L_12_H_768_A_12',
    'cased_L-24_H-1024_A-16': 'cased_L_24_H_1024_A_16',
    'uncased_L-12_H-768_A-12': 'uncased_L_12_H_768_A_12',
    'uncased_L-24_H-1024_A-16': 'uncased_L_24_H_1024_A_16',
    'chinese_L-12_H-768_A-12': 'chinese_L_12_H_768_A_12',
    'multi_cased_L-12_H-768_A-12': 'multi_cased_L_12_H_768_A_12'
}

def download_file(url, filename):
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

def create_hub_bert(model_name, pretrained):
    assert model_name in MODEL_NAME, '{} not in the valid models {}'.format(model_name, MODEL_NAME)
    data_dir = './{}'.format(model_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    vocab_url = '{}/{}/{}'.format(DATA_URL, model_name, VOCAB_NAME)
    config_url = '{}/{}/{}'.format(DATA_URL, model_name, CONFIG_NAME)

    vocab_file = './{}/{}'.format(model_name, VOCAB_NAME)
    config_file = './{}/{}'.format(model_name, CONFIG_NAME)

    download_file(vocab_url, vocab_file)
    download_file(config_url, config_file)

    config = BertConfig(config_file)

    model = hub.load(
        "megengine/models",
        MODEL_NAME[model_name],
        pretrained=pretrained,
    )

    return model, config, vocab_file
```

In order to make it easier for the user to use the pre-trained model, we only keep the `BertModel` part of the original bert model. For example, The` bert` model with pre-trained weights can be used as a part of other models.


```python
class BertForSequenceClassification(Module):
    def __init__(self, config, num_labels, bert):
        self.bert = bert
        self.num_labels = num_labels
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.classifier = Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids,
            attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss = cross_entropy_with_softmax(
                logits.reshape(-1, self.num_labels),
                labels.reshape(-1))
            return logits, loss
        else:
            return logits, None

bert, config, vocab_file = create_hub_bert('uncased_L-12_H-768_A-12', pretrained=True)
model = BertForSequenceClassification(config, num_labels=2, bert=bert)
```

All pre-trained models expect the data to be pre-processed correctly. The requirements are consistent with the Google's bert. For details, please refer to original [bert](https://github.com/google-research/bert), or refer to our example [models](https://github.com/megengine/models/official/nlp/bert).


### Model Description
We provide example code in [models](https://github.com/megengine/models/official/nlp/bert).
This example code fine-tunes the pre-trained `uncased_L-12_H-768_A-12` model on the Microsoft Research Paraphrase (MRPC) dataset.

Our test ran on the original implementation hyper-parameters gave evaluation results between 84% and 88%.


### References
 - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805), Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova;

