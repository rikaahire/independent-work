from gensim.models import KeyedVectors
import numpy as np

# Initialize KeyedVectors object
word_vectors = KeyedVectors(vector_size=3)

# Cosine similarity exactly 0.5
v1 = np.array([1.0, 0.0, 0.0])
v2 = np.array([0.5, np.sqrt(3)/2, 0.0])

# Add the vectors to the KeyedVectors object
word_vectors.add_vector("push", v1)
word_vectors.add_vector("binpick", v2)

# Save the model to a file
word_vectors.save("word2vec.model")
