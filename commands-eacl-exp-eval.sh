exp_dir="pattern-extraction/experiments" 
./translate-subw.sh $exp_dir/italian/bpe1000/gate-sparse-enc-static-head-models $exp_dir/italian/italian-dev $exp_dir/italian/bpe1000/dev_segm 1
./translate-subw.sh $exp_dir/italian/bpe1000/gate-sparse-enc-static-head-models $exp_dir/italian/italian-test $exp_dir/italian/bpe1000/test_segm 1
./translate-subw.sh $exp_dir/italian/bpe1000/gate-sparse-enc-static-head-models $exp_dir/italian/italian-train-high $exp_dir/italian/bpe1000/train_high_segm 1
./translate-subw.sh $exp_dir/finnish/bpe1000/gate-sparse-enc-static-head-models $exp_dir/finnish/finnish-dev $exp_dir/finnish/bpe1000/dev_segm 1
./translate-subw.sh $exp_dir/finnish/bpe1000/gate-sparse-enc-static-head-models $exp_dir/finnish/finnish-test $exp_dir/finnish/bpe1000/test_segm 1
./translate-subw.sh $exp_dir/finnish/bpe1000/gate-sparse-enc-static-head-models $exp_dir/finnish/finnish-train-high $exp_dir/finnish/bpe1000/train_high_segm 1
./translate-subw.sh $exp_dir/tagalog/bpe1000/gate-sparse-enc-static-head-models $exp_dir/tagalog/tagalog-dev $exp_dir/tagalog/bpe1000/dev_segm 1
./translate-subw.sh $exp_dir/tagalog/bpe1000/gate-sparse-enc-static-head-models $exp_dir/tagalog/tagalog-test $exp_dir/tagalog/bpe1000/test_segm 1
./translate-subw.sh $exp_dir/tagalog/bpe1000/gate-sparse-enc-static-head-models $exp_dir/tagalog/tagalog-train-high $exp_dir/tagalog/bpe1000/train_high_segm 1
./translate-ch.sh $exp_dir/italian/ch/gate-sparse-models $exp_dir/italian/italian-dev 1
./translate-ch.sh $exp_dir/italian/ch/gate-sparse-models $exp_dir/italian/italian-test 1
./translate-ch.sh $exp_dir/italian/ch/gate-sparse-models $exp_dir/italian/italian-train-high 1
./translate-ch.sh $exp_dir/italian/ch/gate-sparse-enc-static-head-models $exp_dir/italian/italian-dev 1
./translate-ch.sh $exp_dir/italian/ch/gate-sparse-enc-static-head-models $exp_dir/italian/italian-test 1
./translate-ch.sh $exp_dir/italian/ch/gate-sparse-enc-static-head-models $exp_dir/italian/italian-train-high 1
./translate-ch.sh $exp_dir/finnish/ch/gate-sparse-models $exp_dir/finnish/finnish-dev 1
./translate-ch.sh $exp_dir/finnish/ch/gate-sparse-models $exp_dir/finnish/finnish-test 1
./translate-ch.sh $exp_dir/finnish/ch/gate-sparse-models $exp_dir/finnish/finnish-train-high 1
./translate-ch.sh $exp_dir/finnish/ch/gate-sparse-enc-static-head-models $exp_dir/finnish/finnish-dev 1
./translate-ch.sh $exp_dir/finnish/ch/gate-sparse-enc-static-head-models $exp_dir/finnish/finnish-test 1
./translate-ch.sh $exp_dir/finnish/ch/gate-sparse-enc-static-head-models $exp_dir/finnish/finnish-train-high 1
./translate-ch.sh $exp_dir/tagalog/ch/gate-sparse-models $exp_dir/tagalog/tagalog-dev 1
./translate-ch.sh $exp_dir/tagalog/ch/gate-sparse-models $exp_dir/tagalog/tagalog-test 1
./translate-ch.sh $exp_dir/tagalog/ch/gate-sparse-models $exp_dir/tagalog/tagalog-train-high 1
./translate-ch.sh $exp_dir/tagalog/ch/gate-sparse-enc-static-head-models $exp_dir/tagalog/tagalog-dev 1
./translate-ch.sh $exp_dir/tagalog/ch/gate-sparse-enc-static-head-models $exp_dir/tagalog/tagalog-test 1
./translate-ch.sh $exp_dir/tagalog/ch/gate-sparse-enc-static-head-models $exp_dir/tagalog/tagalog-train-high 1
