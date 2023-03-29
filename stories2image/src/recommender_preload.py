import sys
import os
import numpy as np
import threading
from src.text.encoder import TextEncoder
from src.common.space import Space
from src.config import get_logger

sys.path.append(os.path.join(sys.path[0], 'image', 'captioning'))  # add models.py

logger = get_logger('recommender')

class Recommender:

  def __init__(self, space: Space):
    self.space = space
    self.text_encoder = TextEncoder(space=self.space)
    self._image_subspaces_loc = None

  def set_image_subspaces(self, path: str) -> None:
    self._image_subspaces_loc = path

  def predict(self, genres, text, count: int = 5) -> list:
    print("started prediction...")
    if len(text) == 1:
      sims1 = self.compute_similarities(text, genres)
      return sorted(sims1, key=lambda x: x[1], reverse=True)[:count]
    elif len(text) == 2:
      sims1, sims2 = self.compute_similarities(text, genres)
      return sorted(sims1, key=lambda x: x[1], reverse=True)[:count], sorted(sims2, key=lambda x: x[1], reverse=True)[:count]
    elif len(text) == 3:
      sims1, sims2, sims3 = self.compute_similarities(text, genres)
      return sorted(sims1, key=lambda x: x[1], reverse=True)[:count], sorted(sims2, key=lambda x: x[1], reverse=True)[:count], sorted(sims3, key=lambda x: x[1], reverse=True)[:count]

  def compute_similarities(self, text, genres) -> list:
    print("computing similarities...")

    general_npy = np.load("/home/kryc1/stories2image/data/subspaces_npy/general.npy")
    computer_npy = np.load("/home/kryc1/stories2image/data/subspaces_npy/computer_internet.npy")
    education_npy = np.load("/home/kryc1/stories2image/data/subspaces_npy/education_reference.npy")
    science_npy = np.load("/home/kryc1/stories2image/data/subspaces_npy/science_math.npy")

    sub_txt1 = ""
    sub_txt2 = ""
    sub_txt3 = ""

    sub_txt_total = []

    sims1 = []
    sims2 = []
    sims3 = []

    if len(text) == 1:
      sub_txt1 = self.text_encoder.create_subspace_with_similar(text=text[0])
      sub_txt_total.append(sub_txt1)
    elif len(text) == 2:
      sub_txt1 = self.text_encoder.create_subspace_with_similar(text=text[0])
      sub_txt2 = self.text_encoder.create_subspace_with_similar(text=text[1])
      sub_txt_total.append(sub_txt1)
      sub_txt_total.append(sub_txt2)
    elif len(text) == 3:
      sub_txt1 = self.text_encoder.create_subspace_with_similar(text=text[0])
      sub_txt2 = self.text_encoder.create_subspace_with_similar(text=text[1])
      sub_txt3 = self.text_encoder.create_subspace_with_similar(text=text[2])
      sub_txt_total.append(sub_txt1)
      sub_txt_total.append(sub_txt2)
      sub_txt_total.append(sub_txt3)

    if len(text) == 1:
      subspace_loc = "/home/kryc1/stories2image/data/subspaces/" + genres[0]

      subspaces_npy = ""

      if genres[0] == "science_math":
        subspaces_npy = science_npy
      elif genres[0] == "computer_internet":
        subspaces_npy == computer_npy
      elif genres[0] == "education_reference":
        subspaces_npy = education_npy
      else:
        subspaces_npy = general_npy

      cnt = 0
      print(subspace_loc)

      for im in os.listdir(subspace_loc):
        if cnt % 1000 == 0:
          print(cnt)

        sub_img = subspaces_npy[i]
        sim1 = self.space.subspaces_similarity(sub_txt_total[0], sub_img)
        sims1.append((im.split('.')[0], sim1))

        cnt = cnt + 1

      return sims1
    elif len(text) == 2:
      subspace_loc_1 = "/home/kryc1/stories2image/data/subspaces/" + genres[0]
      subspace_loc_2 = "/home/kryc1/stories2image/data/subspaces/" + genres[1]

      subspaces_npy1 = ""
      subspaces_npy2 = ""

      if genres[0] == "science_math":
        subspaces_npy1 = science_npy
      elif genres[0] == "computer_internet":
        subspaces_npy1 == computer_npy
      elif genres[0] == "education_reference":
        subspaces_npy1 = education_npy
      else:
        subspaces_npy1 = general_npy


      if genres[1] == "science_math":
        subspaces_npy2 = science_npy
      elif genres[1] == "computer_internet":
        subspaces_npy2 == computer_npy
      elif genres[1] == "education_reference":
        subspaces_npy2 = education_npy
      else:
        subspaces_npy2 = general_npy

      cnt = 0
      print(subspace_loc_1)
      print(subspace_loc_2)

      subs1 = os.listdir(subspace_loc_1)
      subs2 = os.listdir(subspace_loc_2)

      subs1.sort()
      subs2.sort()

      sub_len = 0

      if len(subs1) >= len(subs2):
        sub_len = len(subs1)
      else:
        sub_len = len(subs2)

      for i in range(sub_len):
        if cnt % 1000 == 0:
          print(cnt)

        if (i + 1) >= len(subs1):
          pass
        else:
          sub_img1 = subspaces_npy1[i]
          sim1 = self.space.subspaces_similarity(sub_txt_total[0], sub_img1)
          sims1.append((subs1[i].split('.')[0], sim1))
        
        if (i + 1) >= len(subs2):
          pass
        else:
          sub_img2 = subspaces_npy2[i]
          sim2 = self.space.subspaces_similarity(sub_txt_total[1], sub_img2)
          sims2.append((subs2[i].split('.')[0], sim2))

        cnt = cnt + 1

      return sims1, sims2
    elif len(text) == 3:
      subspace_loc_1 = "/home/kryc1/stories2image/data/subspaces/" + genres[0]
      subspace_loc_2 = "/home/kryc1/stories2image/data/subspaces/" + genres[1]
      subspace_loc_3 = "/home/kryc1/stories2image/data/subspaces/" + genres[2]

      subspaces_npy1 = ""
      subspaces_npy2 = ""
      subspaces_npy3 = ""

      if genres[0] == "science_math":
        subspaces_npy1 = science_npy
      elif genres[0] == "computer_internet":
        subspaces_npy1 == computer_npy
      elif genres[0] == "education_reference":
        subspaces_npy1 = education_npy
      else:
        subspaces_npy1 = general_npy


      if genres[1] == "science_math":
        subspaces_npy2 = science_npy
      elif genres[1] == "computer_internet":
        subspaces_npy2 == computer_npy
      elif genres[1] == "education_reference":
        subspaces_npy2 = education_npy
      else:
        subspaces_npy2 = general_npy

      if genres[2] == "science_math":
        subspaces_npy3 = science_npy
      elif genres[2] == "computer_internet":
        subspaces_npy3 == computer_npy
      elif genres[2] == "education_reference":
        subspaces_npy2 = education_npy
      else:
        subspaces_npy2 = general_npy


      cnt = 0
      print(subspace_loc_1)
      print(subspace_loc_2)
      print(subspace_loc_3)

      subs1 = os.listdir(subspace_loc_1)
      subs2 = os.listdir(subspace_loc_2)
      subs3 = os.listdir(subspace_loc_3)

      sub_len = len(subs1)

      if len(subs2) >= sub_len:
        sub_len = len(subs2)
      
      if len(subs3) >= sub_len:
        sub_len = len(subs3)

      for i in range(sub_len):
        if cnt % 1000 == 0:
          print(cnt)

        if (i + 1) >= len(subs1):
          pass
        else:
          sub_img1 = subspaces_npy1
          sim1 = self.space.subspaces_similarity(sub_txt_total[0], sub_img1)
          sims1.append((subs1[i].split('.')[0], sim1))
        
        if (i + 1) >= len(subs2):
          pass
        else:
          sub_img2 = subspaces_npy2
          sim2 = self.space.subspaces_similarity(sub_txt_total[1], sub_img2)
          sims2.append((subs2[i].split('.')[0], sim2))

        if (i + 1) >= len(subs3):
          pass
        else:
          sub_img3 = subspaces_npy3
          sim3 = self.space.subspaces_similarity(sub_txt_total[2], sub_img3)
          sims3.append((subs3[i].split('.')[0], sim3))

        cnt = cnt + 1

      return sims1, sims2, sims3


if __name__ == '__main__':
    from gensim.models import KeyedVectors
    from src.config import Word2VecConfig, ImageConfig

    t_keyed_vectors = KeyedVectors.load_word2vec_format(fname=Word2VecConfig.get_word_vectors_filename(),
                                                        limit=Word2VecConfig.get_vocab_size(),
                                                        binary=True)
    t_space = Space(t_keyed_vectors)
    t_recommender = Recommender(space=t_space)
    t_recommender.set_image_subspaces(path=ImageConfig.get_image_subspaces_folder())
    t_text = "A recently published study analysed concentrations of fine particulate matter pollution across the continental US from 1999 until 2015. Industry, power plants and cars produce these extremely small particles of pollution. They are 30 times smaller than the width of a human hair, and they can be inhaled deep into the lungs, which can lead to a variety of health problems. The study found that this type of pollution declined since 1999, but the researchers say that even at levels below the current standard, air pollution linked to an estimated 30,000 deaths. One of the study’s lead authors said that lowering the standard below the current level would likely improve people’s health."
    preds: list = t_recommender.predict(text=t_text, count=10)
    logger.info(preds)
