## Attention in NLP
Given a set of vector values and vector query, attention computes a weighted sum of the values dependent on the query.
- The weighted sum is a selective summary of the information contained in the values, where the query determines which values to focus on.
- The encoder's hidden state becomes a set of values.
- A learnable relationship between the query and these values signifies the importance of each value concerning the query. 