bpemerges=1000
LangList="italian,finnish,tagalog"
print_commands="yes" #yes/no

Field_Separator=$IFS
IFS=,


for lang in $LangList; do

	#### PREPARE SEGMENTED DATA ####

	# pointers to data files - original (3-columns: lemma features inflected form)
	data_dir=/mnt/eacl/$lang
	train=$data_dir/$lang-train-high
	dev=$data_dir/$lang-dev
	test=$data_dir/$lang-test
	
	# pointer to corpus to train bpe segmentation
	bpe_train_corpus=$data_dir/word-list.txt

	# pointers to paths for saving segmentation model and segented files
	data_dir_bpe=/mnt/eacl/$lang/bpe$bpemerges
	mkdir -p $data_dir_bpe
	train_segm=$data_dir_bpe/train_high_segm
	dev_segm=$data_dir_bpe/dev_segm
	test_segm=$data_dir_bpe/test_segm
	# learn bpe, save model and segmented files to $data_dir/$lang/bpe$bpemerges/
	if [ $print_commands=="yes" ]; then
		echo "./bpe-preprocess.sh $bpemerges $bpe_train_corpus $data_dir_bpe $train $train_segm $dev $dev_segm $test $test_segm"
	else
		./bpe-preprocess.sh $bpemerges $bpe_train_corpus $data_dir_bpe $train $train_segm $dev $dev_segm $test $test_segm
	fi

	#### TRAIN MODELS ####
	
	model_dir=/mnt/eacl/$lang
	##### model with static encoder attenton over subwords
	model_dir_subw=$model_dir/bpe$bpemerges
	mkdir -p $model_dir_subw
	config="config/gate-sparse-enc-static-head.yml"
	if [ $print_commands=="yes" ]; then
		echo "./train-subw.sh $model_dir_subw $train $train_segm $dev $dev_segm $config"
	else
		./train-subw.sh $model_dir_subw $train $train_segm $dev $dev_segm $config
	fi

	###### two char models: sparse two-headed gated attention (baseline) and baseline with static encoder attenton over charcaters
	model_dir_ch=$model_dir/ch
	mkdir -p $model_dir_ch
	ConfigList="config/gate-sparse.yml,config/gate-sparse-enc-static-head.yml"
	for config in $ConfigList; do
		if [ $print_commands=="yes" ]; then
			echo "./train-ch.sh $model_dir_ch $train $dev $config"
		else
			./train-ch.sh $model_dir_ch $train $dev $config
		fi
	done


done

IFS=$Field_Separator