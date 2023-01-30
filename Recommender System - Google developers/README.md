# Recommender System - Google Developers

Tutorial of Basic Recommender System.  
https://developers.google.com/machine-learning/recommendation?hl=ko

# Background

## Terminology

- Items (documents) : The entities a system recommends (apps, videos, etc.)
- Query (context) : The information a system uses to make recommendations.
  - User information
    - id
    - items previously interacted with
  - Additional context
    - time of day
    - device
- Embedding : mapping from discrete set (query, items) to a vector space. Representation of query and items.

## Recommendation Systems Overview

- Candidate Generation
  - Generates small subset of candidates from large corpus (based on query evaluation)
  - A given model may provide multiple candidate generators
  - need quick (probably less precise) evaluation
- Scoring
  - Another model scores and ranks the candidates
  - Can use a more precise model relying on additional queries because of smaller size of candidates
- Re-ranking
  - Final adjustment that can help ensure diversity, freshness and fairness
  - Bonus points for fresh contents, penalty for contents that user explicitly disliked

# Candidate Generation

## Content-based filtering

---

- Uses **similarity between items** to recommend items
- User A watches two cute cat video -> recommend other cute animal video
- _Advantages_
  - No need of any data about other users -> easier to scale to a large number of users
  - Can capture the specific interests of an user
- _Disadvantages_
  - Since feature representation is hand-engineered to some extent, this requires a lot of domain knowledge
  - Limited to users' existing interests

---

### Similarity Measure

A function $s:E\times E->R$ that takes a pair of embeddings and returns a scalar measuring their similarity.  
Most recommendation sytems rely on one or more of the following

- Cosine
  - $s(q,x)=cos(q,x)$
- Dot Product
  - $s(q,x)=<q,x>=\sum_{i=1}^d q_ix_i=|x||q|cos(q,x)$
  - If the embeddings are normalized, dot product and cosine are the same
- Euclidean Distance
  - $s(q,x)=|q-x|=[\sum_{i=1}^d (q_i-x_i)^2]^{1/2}$

---

### Which Similarity Measure to Choose?

Dot product similarity is sensitive to the norm of the embedding vector compared to cosine. Which means that the larger the norm of an embedding, the more likely the item is to be recommended

- Usually, popular items in training set tend to have embeddings with large norms and these items may end up dominating the recommendations. We can put less emphasis on the norm of items by using variants of similarity functions like :  
   _$s(q,x)=|q|^\alpha |x|^\alpha cos(q,x)$_
- Items that appear very raely may not be updated frequently. If we initialize them with a large norm, these rare items would dominate the recommendations. Be careful about **embedding initialization and regularization**!

<br><br/>

## Collaborative filtering

---

- Uses **similarities between queris and items** to provide recommendations
- User A is similar to user B & B likes video1 -> recommend video1 to user A
- _Advantages_
  - No need of domain knowledge
  - Serendipity - can help users discover new interests
  - Great starting point - no need of contextual features. The system needs only the feedback matrix. It can be used as one of multiple candidate generators
- _Disadvantages_
  - Can't handle fresh items (cold-start problem)
    - Projection in WALS - given a new item $i_0$ not seen in training, with few feedback matrix $A_{i_0}$, solve $min_{v_{i_0}}|A_{i_0}-Uv_{i_0}|$
    - Heuristics to generate embeddings of fresh items - approximate by averaging the embeddings of items from same category
  - Hard to include side features (like age or country) for query/item
    - Augment the input matrix with features by defining a block marix $\bar A$
      - Block(0, 0) = original feedback matrix A
      - Block(0, 1) = multi-hot encoding of the user features
      - Block(1, 0) = multi-hot encoding of the item features
      - Block(1, 1) = empty

---

### Matrix Factorization

**Matrix Factorization** is a simple embedding model. Given the feed back matrix $A \in R^{m\times d}$ model learns user embedding matrix $U\in R^{m\times d}$ and item embedding matrix $V\in R^{n\times d}$.  
$O(nm) \rightarrow O((n+m)d)$

---

### Objective Function

- Observed Only MF
  - $\sum_{(i, j)\in obs}(A_{ij}-U_i\cdot V_j)^2$
- Singular Value Decomposition
  - $\sum_{(i, j)}(A_{ij}-U_i\cdot V_j)^2$
  - Not a great solution because matrix A is usualy very sparse. The solution $UV^T$ will likely be close to zero, leading to poor generalization performance
- Weighted Matrix Factorization
  - $\sum_{(i, j)\in obs}w_{i,j}(A_{ij}-U_i\cdot V_j)^2 + w_0\sum_{(i, j)\notin obs}(U_i\cdot V_j)^2$

---

### Minimizing the Objective Function

- SGD (Stochastic Gradient Descent)
  - Very flexible (can use other loss functions)
  - Can be parallelized
  - Slower - takes time to converge
  - Harder to handle the unobserved entries
- WALS (Weighted Alternating Least Squares)
  - Fix U and solve V -> Fix V and solve U -> ...
  - Reliant on Loss Squares only
  - Can be parallelized
  - Converges faster than SGD
  - Easier to handle unobserved entries
