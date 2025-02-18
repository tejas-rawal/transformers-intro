import numpy as np

# Sample word vectors for a text input
word_vectors = np.array([
    [0.2, 0.3, 0.5],
    [0.1, 0.4, 0.6],
    [0.3, 0.2, 0.7]
])

# Compute self-attention weights for each word
attention_weights = np.zeros((len(word_vectors), len(word_vectors)))

for i in range(len(word_vectors)):
    for j in range(len(word_vectors)):
        # Compute the dot product similarity between word vectors
        dot_product_similarity = np.dot(word_vectors[i], word_vectors[j])
        # Use the similarity as the attention weight
        attention_weights[i][j] = dot_product_similarity

# Normalize the attention weights to make them sum to 1
# [:, np.newaxis] is used to reshape the sum vector to a column vector, allowing for element-wise division.
attention_weights = attention_weights / attention_weights.sum(axis=1)[:, np.newaxis]

# Print the attention weights
print("Attention Weights:")
print(attention_weights)