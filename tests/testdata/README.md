This directory contains small "toy" samples drawn from Universal Dependencies
data for English, Greek, and Russian for `udtube_test.py`. The `_train.conllu`
files are used to train and validate the model. The `_expected.conllu`
files are the result of applying the model to the training data, and the
`_expected.test` files give accuracy results. Each file contains ten sentences.

The following commands, run in the root directory, were used to generate the
expected data files:

    udtube fit \
        --config=tests/testdata/udtube_config.yaml \
        --data.model_dir models \
        --data.train=tests/testdata/en_train.conllu \
        --data.val=tests/testdata/en_train.conllu \
        --model.encoder=google-bert/bert-base-cased \
        --model.use_xpos=True
    udtube predict \
        --ckpt_path=models/lightning_logs/version_0/checkpoints/last.ckpt \
        --config=tests/testdata/udtube_config.yaml \
        --data.model_dir=models \
        --data.predict=tests/testdata/en_train.conllu \
        --model.encoder=google-bert/bert-base-cased \
        --model.use_xpos=True \
        > tests/testdata/en_expected.conllu 
    udtube test \
        --ckpt_path=models/lightning_logs/version_0/checkpoints/last.ckpt \
        --config=tests/testdata/udtube_config.yaml \
        --data.model_dir=models \
        --data.test=tests/testdata/en_train.conllu \
        --model.encoder=google-bert/bert-base-cased \
        --model.use_xpos=True \
        > tests/testdata/en_expected.test
    rm -rf models
    udtube fit \
        --config=tests/testdata/udtube_config.yaml \
        --data.model_dir models \
        --data.train=tests/testdata/ru_train.conllu \
        --data.val=tests/testdata/ru_train.conllu \
        --model.encoder=DeepPavlov/rubert-base-cased \
        --model.use_xpos=False
    udtube predict \
        --ckpt_path=models/lightning_logs/version_0/checkpoints/last.ckpt \
        --config=tests/testdata/udtube_config.yaml \
        --data.model_dir=models \
        --data.predict=tests/testdata/ru_train.conllu \
        --model.encoder=DeepPavlov/rubert-base-cased \
        --model.use_xpos=False \
        > tests/testdata/ru_expected.conllu 
    udtube test \
        --ckpt_path=models/lightning_logs/version_0/checkpoints/last.ckpt \
        --config=tests/testdata/udtube_config.yaml \
        --data.model_dir=models \
        --data.test=tests/testdata/ru_train.conllu \
        --model.encoder=DeepPavlov/rubert-base-cased \
        --model.use_xpos=False \
        > tests/testdata/ru_expected.test 
    udtube fit \
        --config=tests/testdata/udtube_config.yaml \
        --data.model_dir models \
        --data.train=tests/testdata/el_train.conllu \
        --data.val=tests/testdata/el_train.conllu \
        --model.encoder=FacebookAI/xlm-roberta-base \
        --model.use_xpos=True
    udtube predict \
        --ckpt_path=models/lightning_logs/version_0/checkpoints/last.ckpt \
        --config=tests/testdata/udtube_config.yaml \
        --data.model_dir=models \
        --data.predict=tests/testdata/el_train.conllu \
        --model.encoder=FacebookAI/xlm-roberta-base \
        --model.use_xpos=True \
        > tests/testdata/el_expected.conllu 
    udtube test \
        --ckpt_path=models/lightning_logs/version_0/checkpoints/last.ckpt \
        --config=tests/testdata/udtube_config.yaml \
        --data.model_dir=models \
        --data.test=tests/testdata/el_train.conllu \
        --model.encoder=FacebookAI/xlm-roberta-base \
        --model.use_xpos=True \
        > tests/testdata/el_expected.test 
    rm -rf models
