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
  part1 = []
  part2 = []
  part3 = []
  part4 = []
  
  part1_2 = []
  part2_2 = []
  part3_2 = []
  part4_2 = []

  part1_3 = []
  part2_3 = []
  part3_3 = []
  part4_3 = []

  def __init__(self, space: Space):
    self.space = space
    self.text_encoder = TextEncoder(space=self.space)
    self._image_subspaces_loc = None

# Thread Func Start ------------------------------------------------------------
  def compute_with_threads(self, partial_list, score_list1, score_list2, score_list3, sub_txt_total):
    print("computing similarities with threads")

    if len(sub_txt_total) == 1:
      cnt = 0
      for im in partial_list:
        sub_img = np.loadtxt(os.path.join(self._image_subspaces_loc, im))
        
        sim1 = self.space.subspaces_similarity(sub_txt_total[0], sub_img)
        score_list1.append((im.split('.')[0], sim1))
        
        cnt += 1
        if(cnt % 1000 == 0):
          print(cnt)
    elif len(sub_txt_total) == 2:
      cnt = 0
      for im in partial_list:
        sub_img = np.loadtxt(os.path.join(self._image_subspaces_loc, im))
        
        sim1 = self.space.subspaces_similarity(sub_txt_total[0], sub_img)
        score_list1.append((im.split('.')[0], sim1))
        
        sim2 = self.space.subspaces_similarity(sub_txt_total[1], sub_img)
        score_list2.append((im.split('.')[0], sim2))

        cnt += 1
        if(cnt % 1000 == 0):
          print(cnt)
    elif len(sub_txt_total) == 3:
      cnt = 0
      for im in partial_list:
        sub_img = np.loadtxt(os.path.join(self._image_subspaces_loc, im))
        
        sim1 = self.space.subspaces_similarity(sub_txt_total[0], sub_img)
        score_list1.append((im.split('.')[0], sim1))
        
        sim2 = self.space.subspaces_similarity(sub_txt_total[1], sub_img)
        score_list2.append((im.split('.')[0], sim2))

        sim3 = self.space.subspaces_similarity(sub_txt_total[2], sub_img)
        score_list3.append((im.split('.')[0], sim3))
        
        cnt += 1
        if(cnt % 100 == 0):
          print(cnt)
# Thread Func End ------------------------------------------------------------

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
    #sub_txt = self.text_encoder.create_subspace(text=text)

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

      cnt = 0
      print(subspace_loc)

      for im in os.listdir(subspace_loc):
        if cnt % 1000 == 0:
          print(cnt)

        sub_img = np.loadtxt(os.path.join(subspace_loc, im))

        sim1 = self.space.subspaces_similarity(sub_txt_total[0], sub_img)
        sims1.append((im.split('.')[0], sim1))

        cnt = cnt + 1

      return sims1
    elif len(text) == 2:
      subspace_loc_1 = "/home/kryc1/stories2image/data/subspaces/" + genres[0]
      subspace_loc_2 = "/home/kryc1/stories2image/data/subspaces/" + genres[1]

      cnt = 0
      print(subspace_loc_1)
      print(subspace_loc_2)

      subs1 = os.listdir(subspace_loc_1)
      subs2 = os.listdir(subspace_loc_2)

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
          sub_img1 = np.loadtxt(os.path.join(subspace_loc_1, subs1[i]))
          sim1 = self.space.subspaces_similarity(sub_txt_total[0], sub_img1)
          sims1.append((subs1[i].split('.')[0], sim1))
        
        if (i + 1) >= len(subs2):
          pass
        else:
          sub_img2 = np.loadtxt(os.path.join(subspace_loc_2, subs2[i]))
          sim2 = self.space.subspaces_similarity(sub_txt_total[1], sub_img2)
          sims2.append((subs2[i].split('.')[0], sim2))

        cnt = cnt + 1

      return sims1, sims2
    elif len(text) == 3:
      subspace_loc_1 = "/home/kryc1/stories2image/data/subspaces/" + genres[0]
      subspace_loc_2 = "/home/kryc1/stories2image/data/subspaces/" + genres[1]
      subspace_loc_3 = "/home/kryc1/stories2image/data/subspaces/" + genres[2]

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
          sub_img1 = np.loadtxt(os.path.join(subspace_loc_1, subs1[i]))
          sim1 = self.space.subspaces_similarity(sub_txt_total[0], sub_img1)
          sims1.append((subs1[i].split('.')[0], sim1))
        
        if (i + 1) >= len(subs2):
          pass
        else:
          sub_img2 = np.loadtxt(os.path.join(subspace_loc_2, subs2[i]))
          sim2 = self.space.subspaces_similarity(sub_txt_total[1], sub_img2)
          sims2.append((subs2[i].split('.')[0], sim2))

        if (i + 1) >= len(subs3):
          pass
        else:
          sub_img3 = np.loadtxt(os.path.join(subspace_loc_3, subs3[i]))
          sim3 = self.space.subspaces_similarity(sub_txt_total[2], sub_img3)
          sims3.append((subs3[i].split('.')[0], sim3))

        cnt = cnt + 1

      return sims1, sims2, sims3
    
    """if self._image_subspaces_loc is not None:
      images_list = os.listdir(self._image_subspaces_loc)
      size = int(len(images_list) / 4)

      t1 = threading.Thread(target=self.compute_with_threads, args=(images_list[:size], self.part1, self.part1_2, self.part1_3 sub_txt_total,))
      t2 = threading.Thread(target=self.compute_with_threads, args=(images_list[size:(2*size)], self.part2, self.part2_2, self.part2_3 sub_txt_total,))
      t3 = threading.Thread(target=self.compute_with_threads, args=(images_list[(2*size):(3*size)], self.part3, self.part3_2, self.part3_3 sub_txt_total,))
      t4 = threading.Thread(target=self.compute_with_threads, args=(images_list[(3*size):], self.part4, self.part4_2, self.part4_3 sub_txt_total,))

      t1.start()
      t2.start()
      t3.start()
      t4.start()

      t1.join()
      t2.join()
      t3.join()
      t4.join()

    sims1 = self.part1 + self.part2 + self.part3 + self.part4
    self.part1 = []
    self.part2 = []
    self.part3 = []
    self.part4 = []
    
    sims2 = self.part1_2 + self.part2_2 + self.part3_2 + self.part4_2
    self.part1_2 = []
    self.part2_2 = []
    self.part3_2 = []
    self.part4_2 = []
    
    sims3 = self.part1_3 + self.part2_3 + self.part3_3 + self.part4_3
    self.part1_3 = []
    self.part2_3 = []
    self.part3_3 = []
    self.part4_3 = []
    
    if len(text) == 1:
      return sims1
    elif len(text) == 2:
      return sims1, sims2
    elif len(text) == 3:
      return sims1, sims2, sims3"""


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