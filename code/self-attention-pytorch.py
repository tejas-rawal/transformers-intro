import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model=2,
                 row_dim=0,
                 col_dim=1):
        """
        @param d_model: dimension of the model, or the number of Word Embedding values per token. Defaults to 2.
        @param row_dim: dimension of the row. Defaults to 0.
        @param col_dim: dimension of the column. Defaults to 1.
        """
        super().__init__()
        # Initialize the Weights matrix (W) that we'll use to create the
        # query (q), key (k) and value (v) for each token.
        # Original Transformer paper did not include bias 
        self.W_q = nn.Linear(in_features=d_model,
                             out_features=d_model,
                             bias=False)
        self.W_k = nn.Linear(in_features=d_model,
                             out_features=d_model,
                             bias=False)
        self.W_v = nn.Linear(in_features=d_model,
                             out_features=d_model,
                             bias=False)
        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encodings):
        """
        Create the query, key and values using the encoding numbers
        associated with each token (token encodings)
        @param token_encodings: torch.Tensor - word embeddings plus positional encoding for each input token
        """
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        # Compute similarities scores a.k.a self-attention: (q * k^T)
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim,
                                          dim1=self.col_dim))
        
        # Scale the similarities by dividing by sqrt(k.col_dim)
        scaled_sims = sims / torch.tensor(k.size(self.col_dim) **0.5)

        # Apply softmax to determine what percent of each tokens' value to
        # use in the final attention values. Determines % of influence each token should have on the others.
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        # Scale the values by their associated percentages and add them up.
        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores


# Testing
# create a matrix of token encodings...
encodings_matrix = torch.tensor([[1.16, 0.23],
                                 [0.57, 1.36],
                                 [4.41, -2.16]])

# set the seed for the random number generator
torch.manual_seed(42)

# create a basic self-attention object
selfAttention = SelfAttention(d_model=2,
                              row_dim=0,
                              col_dim=1)

# calculate basic attention for the token encodings
print(selfAttention(encodings_matrix))
# tensor([[1.0100, 1.0641],
#        [0.2040, 0.7057],
#        [3.4989, 2.2427]], grad_fn=<MmBackward0>)

# print out the weight matrix that creates the queries
print(selfAttention.W_q.weight.transpose(0, 1))
# tensor([[ 0.5406, -0.1657],
#        [ 0.5869,  0.6496]], grad_fn=<TransposeBackward0>)

# print out the weight matrix that creates the keys
print(selfAttention.W_k.weight.transpose(0, 1))
# tensor([[-0.1549, -0.3443],
#        [ 0.1427,  0.4153]], grad_fn=<TransposeBackward0>)

# print out the weight matrix that creates the values
print(selfAttention.W_v.weight.transpose(0, 1))
# tensor([[ 0.6233,  0.6146],
#        [-0.5188,  0.1323]], grad_fn=<TransposeBackward0>)