# Trains sparse two-headed gated attention model with static encoder attenton over subwords

modeldir_base=$1 #pytorch datasets and models will be saved in this folder
train=$2
train_segm=$3
dev=$4
dev_segm=$5

ConfigList="config/gate-sparse-enc-static-head.yml"

####################################################################################################


# prepare datasets
echo "python preprocess.py  -use_bpe -train $train -bpe_train $train_segm -valid $dev -bpe_valid $dev_segm -save_data $modeldir_base/data -inflection_field"
python preprocess.py  -use_bpe -train $train -bpe_train $train_segm -valid $dev -bpe_valid $dev_segm -save_data $modeldir_base/data -inflection_field

# train instruction
Field_Separator=$IFS
IFS=,

for config in $ConfigList; do
    name=$modeldir_base/$( basename $config .yml )
    modeldir=$name-models
    mkdir -p $modeldir
    echo "python train.py -config $config  -data $modeldir_base/data -save_model $modeldir/model -log_file $name.log"
    python train.py -config $config  -data $modeldir_base/data -save_model $modeldir/model -log_file $name.log
done

 
IFS=$Field_Separator

