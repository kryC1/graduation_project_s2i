import torch
import os
from d2l import torch as d2l

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
    cos = torch.mv(W, x.reshape(-1, )) / (
            torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
            torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]

def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')

def get_similar_token(query_token, k, embed=glove_6b100d):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    word = "none"
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        # print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
        word = embed.idx_to_token[int(i)]
    return word