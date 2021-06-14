exp_dir="pattern-extraction/experiments" 
./bpe-preprocess.sh 1000 $exp_dir/italian/word-list.txt $exp_dir/italian/bpe1000 $exp_dir/italian/italian-train-high $exp_dir/italian/bpe1000/train_high_segm $exp_dir/italian/italian-dev $exp_dir/italian/bpe1000/dev_segm $exp_dir/italian/italian-test $exp_dir/italian/bpe1000/test_segm
./train-subw.sh $exp_dir/italian/bpe1000 $exp_dir/italian/italian-train-high $exp_dir/italian/bpe1000/train_high_segm $exp_dir/italian/italian-dev $exp_dir/italian/bpe1000/dev_segm config/gate-sparse-enc-static-head.yml
./train-ch.sh $exp_dir/italian/ch $exp_dir/italian/italian-train-high $exp_dir/italian/italian-dev config/gate-sparse.yml
./train-ch.sh $exp_dir/italian/ch $exp_dir/italian/italian-train-high $exp_dir/italian/italian-dev config/gate-sparse-enc-static-head.yml
./bpe-preprocess.sh 1000 $exp_dir/finnish/word-list.txt $exp_dir/finnish/bpe1000 $exp_dir/finnish/finnish-train-high $exp_dir/finnish/bpe1000/train_high_segm $exp_dir/finnish/finnish-dev $exp_dir/finnish/bpe1000/dev_segm $exp_dir/finnish/finnish-test $exp_dir/finnish/bpe1000/test_segm
./train-subw.sh $exp_dir/finnish/bpe1000 $exp_dir/finnish/finnish-train-high $exp_dir/finnish/bpe1000/train_high_segm $exp_dir/finnish/finnish-dev $exp_dir/finnish/bpe1000/dev_segm config/gate-sparse-enc-static-head.yml
./train-ch.sh $exp_dir/finnish/ch $exp_dir/finnish/finnish-train-high $exp_dir/finnish/finnish-dev config/gate-sparse.yml
./train-ch.sh $exp_dir/finnish/ch $exp_dir/finnish/finnish-train-high $exp_dir/finnish/finnish-dev config/gate-sparse-enc-static-head.yml
./bpe-preprocess.sh 1000 $exp_dir/tagalog/word-list.txt $exp_dir/tagalog/bpe1000 $exp_dir/tagalog/tagalog-train-high $exp_dir/tagalog/bpe1000/train_high_segm $exp_dir/tagalog/tagalog-dev $exp_dir/tagalog/bpe1000/dev_segm $exp_dir/tagalog/tagalog-test $exp_dir/tagalog/bpe1000/test_segm
./train-subw.sh $exp_dir/tagalog/bpe1000 $exp_dir/tagalog/tagalog-train-high $exp_dir/tagalog/bpe1000/train_high_segm $exp_dir/tagalog/tagalog-dev $exp_dir/tagalog/bpe1000/dev_segm config/gate-sparse-enc-static-head.yml
./train-ch.sh $exp_dir/tagalog/ch $exp_dir/tagalog/tagalog-train-high $exp_dir/tagalog/tagalog-dev config/gate-sparse.yml
./train-ch.sh $exp_dir/tagalog/ch $exp_dir/tagalog/tagalog-train-high $exp_dir/tagalog/tagalog-dev config/gate-sparse-enc-static-head.yml
