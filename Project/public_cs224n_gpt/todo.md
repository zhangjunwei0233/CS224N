# paraphrase detection

1. explore how mismatched input format will hurt model performance.

    it appearce not to have significant impacts.

# Sonnet Generation

1. explore fine-tune full model v.s. partial finetune.
    full model finetune worked better.
2. explore if an extra projection layer is needed.
    adding a projection layer actually hards model performance. With the decrease of training set loss, the dev set mark also decrease.
    and the output makes less sense as well.
    Maybe that because the sample size is too small to train a whole new projection layer, which eventually causes overfitting.
