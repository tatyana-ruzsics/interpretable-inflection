# Trains two character-level models: 
#     1. sparse two-headed gated attention (baseline) 
#	  2. baseline with static encoder attenton over charcaters


modeldir_base=$1 #pytorch datasets and models will be saved in this folder
train=$2
dev=$3

ConfigList="config/gate-sparse.yml,config/gate-sparse-enc-static-head.yml"

####################################################################################################


# prepare datasets
#echo "python preprocess.py  -train $train -valid $dev -save_data $modeldir_base/data -inflection_field"
python preprocess.py  -train $train -valid $dev -save_data $modeldir_base/data -inflection_field

# train instruction
Field_Separator=$IFS

IFS=,
for config in $ConfigList; do
    name=$modeldir_base/$( basename $config .yml )
    modeldir=$name-models
    mkdir -p $modeldir
    #echo "python train.py -config $config  -data $datadir/data -save_model $modeldir/model -log_file $name.log"
    python train.py -config $config  -data $datadir/data -save_model $modeldir/model -log_file $name.log
done

IFS=$Field_Separator