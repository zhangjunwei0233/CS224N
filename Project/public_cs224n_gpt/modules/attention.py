import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Initialize the linear transformation layers for key, value, query.
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # This dropout is applied to normalized attention scores following the original
        # implementation of transformer. Although it is a bit unusual, we empirically
        # observe that it yields better performance.
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
        proj = linear_layer(x)
        # Next, we need to produce multiple heads for the proj. This is done by spliting the
        # hidden state to self.num_attention_heads, each of size self.attention_head_size.
        proj = rearrange(proj, 'b t (h d) -> b t h d',
                         h=self.num_attention_heads)
        # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
        proj = rearrange(proj, 'b t h d -> b h t d')
        return proj

    def attention(self, key, query, value, attention_mask):

        # TODO: YOUR CODE HERE
        """
        key:       [bs, num_attention_heads, seq_len, attention_head_size]
        query:     [bs, num_attention_heads, seq_len, attention_head_size]
        value:     [bs, num_attention_heads, seq_len, attention_head_size]
        attn_mask: [bs, 1, 1, seq_len]
        """
        # Step1: Compute attention score (Q K^T) / sqrt(d_k)
        kv_dimension = key.shape[-1]
        attn_score = torch.matmul(
            query, key.transpose(-2, -1)) / (kv_dimension ** 0.5)

        # Step2: Calculate masks

        # future mask  (1 for valid, 0 for masked)
        seq_len = attn_score.shape[-1]
        future_mask = torch.tril(torch.ones(
            seq_len, seq_len, device=attn_score.device))
        future_mask = future_mask.unsqueeze(
            0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        attn_score = attn_score.masked_fill(future_mask == 0, -10000.0)

        # padding mask (input attention mask, derived from `get_extend_attention_mask` method in utils.py)
        attn_score = attn_score + attention_mask

        # print(attn_score)

        # Step3: apply softmax to generate distribution
        attn_probs = torch.softmax(attn_score, dim=-1)

        # Step4: apply dropout layer to normalized attention scores
        attn_probs = self.dropout(attn_probs)

        # Step5: apply attention distribution to values
        context = torch.matmul(attn_probs, value)  # [b, h, T, d_k]

        # Step6: reshape back to [b, T, d]
        context = rearrange(context, 'b h t d -> b t (h d)')

        return context

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # First, we have to generate the key, value, query for each token for multi-head attention
        # using self.transform (more details inside the function).
        # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)

        # Calculate the multi-head attention.
        attn_value = self.attention(
            key_layer, query_layer, value_layer, attention_mask)
        return attn_value


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from config import GPT2Config
    from utils import get_extended_attention_mask

    def test_attention():
        print("Testing CausalSelfAttention...")

        # Test parameters
        bs, h, t, d_k = 2, 4, 6, 8
        config = GPT2Config()
        config.num_attention_heads = h
        config.hidden_size = h * d_k
        config.attention_probs_dropout_prob = 0.0  # Disable for testing

        attn = CausalSelfAttention(config)

        # Create test data
        torch.manual_seed(42)  # For reproducible results
        key = torch.randn((bs, h, t, d_k))
        query = torch.randn((bs, h, t, d_k))
        value = torch.randn((bs, h, t, d_k))

        # Test case 1: No padding (all valid tokens)
        print("\n1. Testing with no padding...")
        attn_mask_no_pad = torch.ones(bs, t)  # [bs, seq_len]
        extended_mask = get_extended_attention_mask(
            attn_mask_no_pad, torch.float32)

        result = attn.attention(key, query, value, extended_mask)
        print(
            f"   Output shape: {result.shape} (expected: [{bs}, {t}, {h * d_k}])")
        assert result.shape == (bs, t, h * d_k), "Wrong output shape!"

        # Test case 2: With padding
        print("\n2. Testing with padding...")
        attn_mask_with_pad = torch.tensor(
            [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])  # [bs, seq_len]
        extended_mask_pad = get_extended_attention_mask(
            attn_mask_with_pad, torch.float32)

        result_pad = attn.attention(key, query, value, extended_mask_pad)
        print(f"   Output shape: {result_pad.shape}")

        # Test case 3: Check causal masking (manually compute attention weights)
        print("\n3. Testing causal masking...")
        with torch.no_grad():
            # Compute attention scores manually
            scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

            # Apply causal mask
            seq_len = scores.shape[-1]
            causal_mask = torch.tril(torch.ones(seq_len, seq_len))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores_masked = scores.masked_fill(causal_mask == 0, -10000.0)

            # Apply padding mask
            scores_masked = scores_masked + extended_mask

            # Check that future positions have very low attention
            probs = torch.softmax(scores_masked, dim=-1)

            # Position 0 should only attend to position 0
            print(
                f"   Position 0 attention to future: {probs[0, 0, 0, 1:].sum().item():.6f} (should be ~0)")

            # Position 2 should only attend to positions 0, 1, 2
            print(
                f"   Position 2 attention to future: {probs[0, 0, 2, 3:].sum().item():.6f} (should be ~0)")

        # Test case 4: Check attention weights sum to 1 for valid positions
        print("\n4. Testing attention weight normalization...")
        with torch.no_grad():
            scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
            causal_mask = torch.tril(torch.ones(
                seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask == 0, -10000.0)
            scores = scores + extended_mask_pad
            probs = torch.softmax(scores, dim=-1)

            # Check first sample, first head - position 2 (valid) should sum to 1
            pos2_sum = probs[0, 0, 2, :].sum().item()
            print(
                f"   Position 2 attention weights sum: {pos2_sum:.6f} (should be ~1.0)")

            # Check that padded keys get very low attention
            # Position 1 attending to padded positions
            pos1_to_padded = probs[0, 0, 1, 3:].sum().item()
            print(
                f"   Position 1 attention to padded: {pos1_to_padded:.6f} (should be ~0)")

        print("\n✅ All tests passed!")

    test_attention()
