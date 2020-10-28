# Train BPE model and segment train, dev, test sets

bpemerges=$1
bpe_train_corpus=$2
bpe_model_folder=$3
train=$4
train_segm=$5
dev=$6
dev_segm=$7
test=$8
test_segm=$9

####################################################################################################

# train bpe model
#echo "tools/learn_bpe.py -s $bpemerges < $bpe_train_corpus > $bpe_model_folder/bpe-codes.src"
tools/learn_bpe.py -s $bpemerges < $bpe_train_corpus > ${bpe_model_folder}/bpe-codes.src

# apply bpe model

#echo "cut -f1 $train | tools/apply_bpe.py -c $bpe_model_folder/bpe-codes.src |sed -r 's/(@@ )|(@@ ?$)/|/g' > $train_segm"
cut -f1 $train | tools/apply_bpe.py -c $bpe_model_folder/bpe-codes.src |sed -r 's/(@@ )|(@@ ?$)/|/g' > $train_segm

#echo "cut -f1 $dev | tools/apply_bpe.py -c $bpe_model_folder/bpe-codes.src |sed -r 's/(@@ )|(@@ ?$)/|/g' > $dev_segm"
cut -f1 $dev | tools/apply_bpe.py -c $bpe_model_folder/bpe-codes.src |sed -r 's/(@@ )|(@@ ?$)/|/g' > $dev_segm

#echo "cut -f1 $test| tools/apply_bpe.py -c $bpe_model_folder/bpe-codes.src |sed -r 's/(@@ )|(@@ ?$)/|/g' > $test_segm"
cut -f1 $test | tools/apply_bpe.py -c $bpe_model_folder/bpe-codes.src |sed -r 's/(@@ )|(@@ ?$)/|/g' > $test_segm
