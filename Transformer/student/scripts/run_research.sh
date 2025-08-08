##! /bin/bash

# Pretrain the model
python src/run.py pretrain research wiki.txt \
        --writing_params_path research.pretrain.params
        
# Finetune the model
python src/run.py finetune research wiki.txt \
        --reading_params_path research.pretrain.params \
        --writing_params_path research.finetune.params \
        --finetune_corpus_path birth_places_train.tsv
        
# Evaluate on the dev set; write to disk
python src/run.py evaluate research wiki.txt  \
        --reading_params_path research.finetune.params \
        --eval_corpus_path birth_dev.tsv \
        --outputs_path research.pretrain.dev.predictions
        
# Evaluate on the test set; write to disk
python src/run.py evaluate research wiki.txt  \
        --reading_params_path research.finetune.params \
        --eval_corpus_path birth_test_inputs.tsv \
        --outputs_path research.pretrain.test.predictions
