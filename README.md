# Neural seq2seq tools for morphological processing
This repo contains seq2seq models for morphological processing, adapted to the task of morphological inflection.
The system is implemented in [PyTorch](https://github.com/pytorch/pytorch) with a codebase derived from [OpenNMT-py](https://github.com/deep-spin/OpenNMT-entmax) adaptation for this task at http://github.com/deep-spin/SIGMORPHON2019. 

## Supported Models
* chED: character-level encoder-decoder model with sparse two-headed gated attention (see [Peters and Martins, 2019 ACL](https://www.aclweb.org/anthology/W19-4207/))
* chED+subwSELF-ATT: chED model with additional _self-attention_ mechanism over _subwords_
* chED+chSELF-ATT: chED model with additional _self-attention_ mechanism over _characters_

![Image of chED+subwSELF-ATT](https://github.com/tatyana-ruzsics/interpretable-inflection/blob/main/subw-self-att.png)

## Installation
Install pytorch v>=1.2 and additional python requirments:
```shell-session
pip install -r requirements.txt
```

## Training models
Use provided `train-subw.sh` file to train subword-level model (chED+subwSELF-ATT) and `train-ch.sh` file to train character-level models (chED, chED+chSELF-ATT). Both files require paths to model, train and development files as arguments. In the subword-level setting, aditional segmented form of the data is required which can be produced with a provided script `bpe-preprocess.sh`. An example of training all three models:

```shell-session
 
#### PREPARE SEGMENTED DATA ####
bpemerges=1000

# pointers to data files - original (3-columns: lemma features inflected form)
train=path/to/train/file
dev=path/to/dev/file
test=path/to/test/file
	
# pointer to corpus to train BPE segmentation
bpe_train_corpus=path/to/BPE/corpus

# pointers to paths for saving segmentation model and segmented files
data_dir_bpe=path/to/save/BPE/model
train_segm==path/to/save/segmented/train/file
dev_segm=path/to/save/segmented/dev/file
test_segm=path/to/save/segmented/test/file

# Train BPE model, save model and segmented files to $data_dir/$lang/bpe$bpemerges/
./bpe-preprocess.sh $bpemerges $bpe_train_corpus $data_dir_bpe $train $train_segm $dev $dev_segm $test $test_segm

#### TRAIN MODELS ####

##### subword-level model (chED+subwSELF-ATT) #####
model_dir_subw=path/to/save/model
./train-subw.sh $model_dir_subw $train $train_segm $dev $dev_segm

###### character-level models (chED, chED+chSELF-ATT) #####
model_dir_ch=path/to/save/model
./train-ch.sh $model_dir_ch $train $dev
```
## Evaluating models
Use provided `translate-subw.sh` file to train subword-level model (chED+subwSELF-ATT) and `translate-ch.sh` file to train character-level models (chED, chED+chSELF-ATT). Both files require paths to pretrained models and test file as arguments. In the subword-level setting, aditional segmented form of the data is required. An example of running evaluation all three models:

```shell-session
beam=1

# pointers to test data - original (3-columns: lemma features inflected form)
test=path/to/test/file

# pointers to segmented test data
test_segm=path/to/segmented/test/file

##### subword-level model (chED+subwSELF-ATT) #####
# pointers to pretrained models
modeldir=/path/to/models
./translate-subw.sh $modeldir $test $test_segm $beam

##### character-level models (chED, chED+chSELF-ATT)  #####
# pointers to pretrained models
modeldir=/path/to/models
./translate-ch.sh $modeldir $test $beam 
```
