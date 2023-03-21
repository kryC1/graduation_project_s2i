import numpy as np
from collections import Counter
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles


class Space:

  def __init__(self, keyed_vectors: KeyedVectors):
    self._keyed_vectors = keyed_vectors
    self._embed_size: int = 300

  def get_embedding(self, word: str) -> np.ndarray:
    """ What is the vector for missing words? """
    return self._keyed_vectors[word] if word in self._keyed_vectors else None

  def create_subspace(self, keywords: Counter, dim: int, use_tf: bool = True) -> np.ndarray:
    print("create_subspace (space.py)")
    embeddings = self._vectorize_keywords(keywords, use_tf)
    print(embeddings)
    R = self._autocorr(embeddings)
    return PCA(n_components=dim).fit_transform(R)

  def _vectorize_keywords(self, keywords: Counter, use_tf: bool = True) -> np.ndarray:
    print("vectorizing keywords...")
    if not use_tf:
      print("vectorizing first")
      return np.array([
        self.get_embedding(word)
        for word in keywords.keys()
        if self.get_embedding(word) is not None
      ])
    print("vectorizing second")
    return np.array([
      self.get_embedding(word) * count
      for word, count in keywords.items()
      if self.get_embedding(word) is not None
    ])

  @staticmethod
  def subspaces_similarity(S1, S2):
    canon_angles = subspace_angles(S1, S2)
    return np.average(np.square(np.cos(canon_angles)))

  @staticmethod
  def _autocorr(matrix) -> np.ndarray:
    return matrix.T.dot(matrix)


if __name__ == '__main__':
    from src.config import Word2VecConfig

    t_keyed_vectors = KeyedVectors.load_word2vec_format(fname=Word2VecConfig.get_word_vectors_filename(),
                                                        limit=Word2VecConfig.get_vocab_size(),
                                                        binary=True)
    t_space = Space(t_keyed_vectors)
