from collections import Counter
from src.common.basic_encoder import BasicEncoder
from src.common.space import Space
from src.text.summarizer import NewsSummarizer
from src.config import StopWordsConfig
import os
import torch

#-------------------------------------------
from torch import nn
from d2l import torch as d2l

d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip', 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')
#-------------------------------------------------

class TokenEmbedding:

    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(embedding_name)
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}
        self.unknown_idx = 0

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        
        with open(os.path.join(data_dir, 'vec.txt'), 'r', encoding="utf8") as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]

                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)

glove_6b100d = TokenEmbedding('glove.6b.100d')

def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]

def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    word = "none"
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
        word = embed.idx_to_token[int(i)]
    
    return word


class TextEncoder(BasicEncoder):

  def __init__(self, space: Space):
    super().__init__(space)
    self.summarizer = NewsSummarizer(stopwords=StopWordsConfig.get_stopwords())

  def create_subspace(self, text: str, dim: int = 5):
    print("creating subspace...")
    """Returns a word subspace """
    keywords: Counter = self._get_keywords(text=text)

    if keywords:
      print("returning keywords...")
      return self.space.create_subspace(keywords=keywords, dim=dim)

  def create_subspace_with_similar(self, text: str, dim: int = 5):
    print("creating subspace...")
    """Returns a word subspace """
    keywords: Counter = self._get_keywords(text=text)
    keywords_new = Counter()

    for i in keywords:
      new_word = get_similar_tokens(i, 1, glove_6b100d)
      keywords_new[new_word] = 1

    print(keywords)
    print(keywords_new)

    merged_keywords = keywords + keywords_new
    
    if merged_keywords:
      print("returning keywords...")
      return self.space.create_subspace(keywords=merged_keywords, dim=dim)

  def _get_keywords(self, text: str, limit: int = 5) -> Counter:
    print("getting keywords...")
    sentence_limit = 1
    keywords = Counter()
    while not keywords:
        keywords: Counter = self.summarizer.get_keywords(text=text, sentence_limit=sentence_limit, keyword_limit=limit)
        sentence_limit += 1
    print(f'Encoder. Keywords: {keywords}')
    return keywords


if __name__ == '__main__':
    from gensim.models import KeyedVectors
    from src.config import Word2VecConfig

    t_keyed_vectors = KeyedVectors.load_word2vec_format(fname=Word2VecConfig.get_word_vectors_filename(),
                                                        limit=Word2VecConfig.get_vocab_size(),
                                                        binary=True)
    t_space = Space(t_keyed_vectors)
    text_encoder = TextEncoder(space=t_space)

    t_text = \
        "Shocking CCTV footage released by Manchester police shows \
        the moment the man wielding a large-bladed knife is tackled \
        to the ground by armed officers. \
        At about 11 pm on Tuesday, CCTV operators spotted a man \
        waving the butcher’s knife around the Piccadilly Garden’s \
        area of Manchester and informed the police. \
        The man can be seen struggling to stand and interacts with \
        terrified members of the public, as he continues to wave \
        the knife around. \
        A 55-year-old man has been arrested on \
        suspicion of affray and remains in police custody for questioning."
    t_keywords = text_encoder._get_keywords(text=t_text)
    print(t_keywords)