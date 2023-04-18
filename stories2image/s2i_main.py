"""import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')"""

import sys
import os
sys.path.append('../..')
sys.path.append("src/image/captioning")  # add models.py
from typing_extensions import TypedDict
from gensim.models import KeyedVectors
from src.text.summarizer import NewsSummarizer
from src.image.encoder import ImageEncoder
from src.image.image_grouping import ImageGrouping
#from src.recommender import Recommender, Space
#from src.recommender_threads import Recommender, Space
from src.recommender_preload import Recommender, Space
from src.config import Word2VecConfig, ImageConfig
from src.similar_words import get_similar_token
from src.paragraph_split import split_paragraph

# TEXT CLASSIFICATION LIBRARY - START
import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
import pickle
# TEXT CLASSIFICATION LIBRARY - END


keyed_vectors = KeyedVectors.load_word2vec_format(fname=Word2VecConfig.get_word_vectors_filename(),
                                                  limit=Word2VecConfig.get_vocab_size(),
                                                  binary=True)
space = Space(keyed_vectors)
recommender = Recommender(space=space)
recommender.set_image_subspaces(path=ImageConfig.get_image_subspaces_folder())
images_folder = ImageConfig.get_images_folder()
summarizer = NewsSummarizer()
image_encoder = ImageEncoder(space=space)
image_grouping = ImageGrouping(n_clusters=4)


# TEXT CLASSIFICATION STARTS FROM HERE
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
                 2: "science_math",
                 3: "Health",
                 4: "education_reference",
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
# TEXT CLASSIFICATION ENDS HERE


# MAIN STARTS HERE
class Stories2Image:
  images_folder = ImageConfig.get_images_folder()

  def __init__(self, recommender: Recommender, summarizer: NewsSummarizer,
                image_encoder: ImageEncoder, image_grouping: ImageGrouping):
    self.recommender = recommender
    self.summarizer = summarizer
    self.image_encoder = image_encoder
    self.image_grouping = image_grouping

  def process(self, text: str):
    pass

  def _get_text_summary(self, text: str) -> str:
    return self.summarizer.generate_summary(text=text)

  def _get_text_keywords(self, text: str) -> set:
    return set(self.summarizer.get_keywords(text=text))

  def _get_recommended_image_ids(self, text: str, count: int = 5) -> list:
    images_rec: list = self.recommender.predict(text=text, count=count)
    _ids = [img[0] for img in images_rec]
    print(f'Selected image IDs: {_ids}')
    return _ids

  def _get_recommended_image_ids_list(self, genres, text: str, count: int = 5) -> list:
    if len(text) == 1:
      images_rec1 = self.recommender.predict(genres, text=text, count=count)
      _ids1 = [img[0] for img in images_rec1]
      print(_ids1)
      return _ids1
    elif len(text) == 2:
      images_rec1, images_rec2 = self.recommender.predict(genres, text=text, count=count)
      _ids1 = [img[0] for img in images_rec1]
      _ids2 = [img[0] for img in images_rec2]
      print(_ids1)
      print(_ids2)
      return _ids1, _ids2
    elif len(text) == 3:
      images_rec1, images_rec2, images_rec3 = self.recommender.predict(genres, text=text, count=count)
      _ids1 = [img[0] for img in images_rec1]
      _ids2 = [img[0] for img in images_rec2]
      _ids3 = [img[0] for img in images_rec3]
      print(_ids1)
      print(_ids2)
      print(_ids3)
      return _ids1, _ids2, _ids3

  def _get_selected_images(self, text: str):
    rec_ids = self._get_recommended_image_ids(text=text)
    img_fns: list = [os.path.join(self.images_folder, f'{_id}.jpg') for _id in rec_ids]
    reps: list = self.image_grouping.get_representatives(image_filenames=img_fns)
    return reps

  def _get_selected_images_list(self, text_list, genres):
    if len(text_list) == 1:
      rec_ids1 = self._get_recommended_image_ids_list(genres, text_list)
      img_fns1: list = [os.path.join(self.images_folder + genres[0] + "/", f'{_id}.jpg') for _id in rec_ids1]
      reps1: list = self.image_grouping.get_representatives(image_filenames=img_fns1)
      return reps1
    elif len(text_list) == 2:
      rec_ids1, rec_ids2 = self._get_recommended_image_ids_list(genres, text_list)
      img_fns1: list = [os.path.join(self.images_folder + genres[0] + "/", f'{_id}.jpg') for _id in rec_ids1]
      img_fns2: list = [os.path.join(self.images_folder + genres[1] + "/", f'{_id}.jpg') for _id in rec_ids2]
      reps1: list = self.image_grouping.get_representatives(image_filenames=img_fns1)
      reps2: list = self.image_grouping.get_representatives(image_filenames=img_fns2)
      return reps1, reps2
    elif len(text_list) == 3:
      rec_ids1, rec_ids2, rec_ids3 = self._get_recommended_image_ids_list(genres, text_list)
      img_fns1: list = [os.path.join(self.images_folder + genres[0] + "/", f'{_id}.jpg') for _id in rec_ids1]
      img_fns2: list = [os.path.join(self.images_folder + genres[1] + "/", f'{_id}.jpg') for _id in rec_ids2]
      img_fns3: list = [os.path.join(self.images_folder + genres[2] + "/", f'{_id}.jpg') for _id in rec_ids3]
      reps1: list = self.image_grouping.get_representatives(image_filenames=img_fns1)
      reps2: list = self.image_grouping.get_representatives(image_filenames=img_fns2)
      reps3: list = self.image_grouping.get_representatives(image_filenames=img_fns3)
      return reps1, reps2, reps3

  def _get_image_captions(self):
    pass

  def _get_image_keywords(self):
    pass


def get_image_ids(path_list, genre):
  id_list = []

  for path in path_list:
    splitted = path.split('/')
    id_list.append("/home/kryc1/s2i_django/stories2image/static/images/" + genre + "/" + splitted[8])
  
  return id_list


def start_process(text_raw, order_input):
  #text_raw1 = "There were a boy and girl. They were at the beach. They were enjoying the sea and the sun. The girl suddenly got sick and wanted to leave. She took her white bag and left."
  splitted_list = []
  summary_list = []
  keywords_org_list = []
  keywords_new_list = []
  genres_list = []
  image_path_list = []

  text_splitted = split_paragraph(text_raw, order_input).split("\n")

  gen1 = "general"
  gen2 = "general"
  gen3 = "general"

  genres = []

  if len(text_splitted) == 1:
    gen1 = predict(text_splitted[0])

    if gen1 == "family_relationships" or gen1 == "Entertainment & Music" or gen1 == "Sports" or "education_reference":
      genres.append("general")
    else:
      genres.append(gen1)

  elif len(text_splitted) == 2:
    gen1 = predict(text_splitted[0])
    gen2 = predict(text_splitted[1])

    if gen1 == "family_relationships" or gen1 == "Entertainment & Music" or gen1 == "Sports" or "education_reference":
      genres.append("general")
    else:
      genres.append(gen1)

    if gen2 == "family_relationships" or gen2 == "Entertainment & Music" or gen2 == "Sports" or "education_reference":
      genres.append("general")
    else:
      genres.append(gen2)

  elif len(text_splitted) == 3:
    gen1 = predict(text_splitted[0])
    gen2 = predict(text_splitted[1])
    gen3 = predict(text_splitted[2])

    if gen1 == "family_relationships" or gen1 == "Entertainment & Music" or gen1 == "Sports" or "education_reference":
      genres.append("general")
    else:
      genres.append(gen1)

    if gen2 == "family_relationships" or gen2 == "Entertainment & Music" or gen2 == "Sports" or "education_reference":
      genres.append("general")
    else:
      genres.append(gen2)

    if gen3 == "family_relationships" or gen3 == "Entertainment & Music" or gen3 == "Sports" or "education_reference":
      genres.append("general")
    else:
      genres.append(gen3)


  model = Stories2Image(recommender=recommender, summarizer=summarizer,
                     image_encoder=image_encoder, image_grouping=image_grouping)

  genre_cnt = 0


  for i in text_splitted:
    splitted_list.append(i)

    try:
      summary_list.append(i)
    except:
      summary_list.append(model._get_text_summary(text=i))

    keywords_org = model._get_text_keywords(text=i)
    keywords_org_list.append(keywords_org)

    keywords_new = []
    for i in keywords_org:
      keywords_new.append(get_similar_token(i, 1))
    keywords_new_list.append(keywords_new)

    genres_list.append(genres[genre_cnt])
    genre_cnt += 1

  if len(text_splitted) == 1:
    reps_to_print1 = model._get_selected_images_list(text_splitted, genres)
    image_path_list.append(get_image_ids(reps_to_print1, genres[0]))
  elif len(text_splitted) == 2:
    reps_to_print1, reps_to_print2 = model._get_selected_images_list(text_splitted, genres)
    image_path_list.append(get_image_ids(reps_to_print1, genres[0]))
    image_path_list.append(get_image_ids(reps_to_print2, genres[1]))
  elif len(text_splitted) == 3:
    reps_to_print1, reps_to_print2, reps_to_print3 = model._get_selected_images_list(text_splitted, genres)
    image_path_list.append(get_image_ids(reps_to_print1, genres[0]))
    image_path_list.append(get_image_ids(reps_to_print2, genres[1]))
    image_path_list.append(get_image_ids(reps_to_print3, genres[2]))

  print("process complete")
  return splitted_list, image_path_list, genres_list, keywords_new_list, keywords_org_list, summary_list