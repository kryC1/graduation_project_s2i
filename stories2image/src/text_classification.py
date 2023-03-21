from torch import nn
from torchtext.data.utils import get_tokenizer
import torch
import pickle

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

with open('/home/kryc1/stories2image/data/text_classifier', 'rb') as training_model:
    class_model = pickle.load(training_model)

with open('/home/kryc1/stories2image/data/vocab', 'rb') as vocab:
    vocab = pickle.load(vocab)

tokenizer = get_tokenizer('basic_english')
text_pipeline = lambda x: vocab(tokenizer(x))
class_model = class_model.to("cpu")

ag_labels = {1: "Society & Culture",
                 2: "Science & Mathematics",
                 3: "Health",
                 4: "Education & Reference",
                 5: "Computers & Internet",
                 6: "Sports",
                 7: "Business & Finance",
                 8: "Entertainment & Music",
                 9: "family_relationships",
                 10: "Politics & Government",
                 }

def predict(text, text_pipeline = text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = class_model(text, torch.tensor([0]))
        return ag_labels[output.argmax(1).item() + 1]