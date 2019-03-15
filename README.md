# CGED_TF

#### 整体思路

结合比赛本身提供的数据，可以将该问题建模为一个标注问题，例如命名实体识别等。NER的经典方案是BiLSTM+CRF。

#### 模型架构

![img](http://wx4.sinaimg.cn/mw690/aba7d18bgy1g13ty32x8fj20md0bmtbp.jpg)

#### 特征

输入特征为四个，分为三类:

(1) bigram特征，包括当前和前一时间步的bigram特征 

(2) pos特征: 特征提取工具使用哈工大的LTP，任何其他开源工具都可以，目的是做词性标注

(3) char特征

#### 围绕该工作，我的相关博客

[中文拼写纠错-和百度比一比](https://zhpmatrix.github.io/2019/02/01/rethinking-spellchecker/),这篇博客主要和百度AI平台开放的中文纠错API的结果对比，同时给出了如何接入API的方法。

[中文拼写纠错](https://zhpmatrix.github.io/2018/12/17/chinese-spell-checker/)，这篇博客简单的梳理了围绕该课题做的相关工作。


