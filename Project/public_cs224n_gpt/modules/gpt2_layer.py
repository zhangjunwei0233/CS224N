from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention


class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Multi-head attention.
        self.self_attention = CausalSelfAttention(config)
        # Add-norm for multi-head attention.
        self.attention_dense = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Feed forward.
        self.interm_dense = nn.Linear(
            config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # Add-norm for feed forward.
        self.out_dense = nn.Linear(
            config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add(self, input, output, dense_layer, dropout):
        """
        TODO: Implement this helper method for the forward function.
          - This function is applied after the multi-head attention layer as well as after the feed forward layer.
          - GPT-2 layer applies dropout to the transformed output of each sub-layer,
            before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
            IN THIS FUNCTION.
        """
        # TODO: YOUR CODE HERE

        # Step1: dense output layer
        dense_output = dense_layer(output)

        # Step2: apply dropout to densed output
        dropout_output = dropout(dense_output)

        # Step3: add input to output
        return input + dropout_output

    def forward(self, hidden_states, attention_mask):
        """
        TODO: Implement the forward pass. Some key points to consider:
               - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
               - Layer normalization applied *before* the attention layer and feed-forward layer.
               - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
               - A feed-forward layer that applies transformations to further refine the hidden states.
        """

        # TODO: YOUR CODE HERE

        # Step1: apply normalization
        normalized_input = self.attention_layer_norm(hidden_states)

        # Step2: hidden_states go through multi-head attention
        attention_output = self.self_attention.forward(
            normalized_input, attention_mask)

        # Step3: add & drop
        processed_attention_output = self.add(
            hidden_states, attention_output, self.attention_dense, self.attention_dropout)

        # Step4: apply normalization
        normalized_output = self.out_layer_norm(processed_attention_output)

        # Step5: processed output go through feed-forward layer
        ff_middle = self.interm_dense(normalized_output)
        ff_output = self.interm_af(ff_middle)

        # Step6: add & drop
        final_output = self.add(
            processed_attention_output, ff_output, self.out_dense, self.out_dropout)

        return final_output
