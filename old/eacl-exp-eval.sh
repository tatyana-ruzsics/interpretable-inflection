

bpemerges=1000
beam=1
print_commands="yes" #yes/no

Field_Separator=$IFS
IFS=,

# evaluate subword model
LangList="italian,finnish,tagalog"

ModelList="gate-sparse-enc-static-head"

for lang in $LangList; do
	for model in $ModelList; do

		# pointers to data files - original (3-columns: lemma features inflected form)
		data_dir=/mnt/eacl/$lang
		train=$data_dir/$lang-train-high
		dev=$data_dir/$lang-dev
		test=$data_dir/$lang-test

		# pointers to paths for segmented files
		data_dir_bpe=/mnt/eacl/$lang/bpe$bpemerges
		train_segm=$data_dir_bpe/train_high_segm
		dev_segm=$data_dir_bpe/dev_segm
		test_segm=$data_dir_bpe/test_segm
		
		modeldir=/mnt/eacl/$lang/bpe$bpemerges/$model-models
		if [ $print_commands=="yes" ]; then
			echo "./translate-subw.sh $modeldir $dev $dev_segm $beam"
			echo "./translate-subw.sh $modeldir $test $test_segm $beam"
			echo "./translate-subw.sh $modeldir $train $train_segm $beam"
		else
			./translate-subw.sh $modeldir $dev $dev_segm $beam
			./translate-subw.sh $modeldir $test $test_segm $beam
			./translate-subw.sh $modeldir $train $train_segm $beam
		fi

	done
done


# evaluate char models
LangList="italian,finnish,tagalog"
ModelList="gate-sparse,gate-sparse-enc-static-head"

for lang in $LangList; do
	for model in $ModelList; do

		# pointers to data files - original (3-columns: lemma features inflected form)
		data_dir=/mnt/eacl/$lang
		train=$data_dir/$lang-train-high
		dev=$data_dir/$lang-dev
		test=$data_dir/$lang-test

		modeldir=/mnt/eacl/$lang/ch/$model-models
		if [ $print_commands=="yes" ]; then
			echo "./translate-ch.sh $modeldir $dev $beam"
			echo "./translate-ch.sh $modeldir $test $beam"
			echo "./translate-ch.sh $modeldir $train $beam"
		else
			./translate-ch.sh $modeldir $dev $beam
			./translate-ch.sh $modeldir $test $beam
			./translate-ch.sh $modeldir $train $beam
		fi
	done
done

IFS=$Field_Separator

