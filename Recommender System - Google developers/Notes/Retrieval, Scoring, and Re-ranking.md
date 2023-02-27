# Retrieval

**Suppose you have an embedding model. Given a user, ow would you decide which items to recommend?**

- For a Matrix Factorization model, the user(query) embedding is known. The system can simply look it up from the user embedding matrix.
- For a DNN model, the system will to compute the user(query) embedding $\psi(x)$ at serve time based on the given features $x$.

Once you have the query embedding, you can decide nearest neighbors by similarity scores $s(q, V_j$) between the query embedding $q$ and item embeddings $V_j$.

In related-item recommendations, you can apply the same approach. First look up the embedding of the item, and then compare it with other item embeddings to find top-k nearest neighbors.

### Large-scale Retrieval

To find the nearest neighbors, the system need to compute similarity score of all possible candidates. This exhaustive scoring can be very expensive for large corpora. The following strategies can be helpful.

- If the query embedding is know statically, perform exhaustive scoring offline (before online service). This is a common practive for related-item recommendation
- Use approximate nearest neighbors

# Scoring
