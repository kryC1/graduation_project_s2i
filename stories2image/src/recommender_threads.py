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

# Thread Func Start ------------------------------------------------------------
  def inside_thread_func(self, subspace_location, partial_txt_list, input_subspace, partial_score_list):
    cnt = 0

    for im in partial_txt_list:
      if cnt % 1000 == 0:
        print(cnt)

      sub_img = np.loadtxt(os.path.join(subspace_location, im))

      sim1 = self.space.subspaces_similarity(input_subspace, sub_img)
      partial_score_list.append((im.split('.')[0], sim1))

      cnt = cnt + 1


  def search_compare_thread(self, subspace_location, input_subspace, score_list):
    part1 = []
    part2 = []
    part3 = []
    part4 = []

    loc_main = os.listdir(subspace_location)

    size = int(len(loc_main))

    th1 = threading.Thread(target=self.inside_thread_func, args=(subspace_location, loc_main[:size], input_subspace, part1,))
    th2 = threading.Thread(target=self.inside_thread_func, args=(subspace_location, loc_main[size: 2*size], input_subspace, part2,))
    th3 = threading.Thread(target=self.inside_thread_func, args=(subspace_location, loc_main[2*size : 3*size], input_subspace, part3,))
    th4 = threading.Thread(target=self.inside_thread_func, args=(subspace_location, loc_main[3*size:], input_subspace, part4,))

    th1.start()
    th2.start()
    th3.start()
    th4.start()

    th1.join()
    th2.join()
    th3.join()
    th4.join()

    score_list.extend(part1)
    score_list.extend(part2)
    score_list.extend(part3)
    score_list.extend(part4)
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
    print("computing similarities wtih threads...")
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

      print(subspace_loc_1)
      print(subspace_loc_2)

      t1 = threading.Thread(target=self.search_compare_thread, args=(subspace_loc_1, sub_txt_total[0], sims1,))
      t2 = threading.Thread(target=self.search_compare_thread, args=(subspace_loc_2, sub_txt_total[1], sims2,))

      t1.start()
      t2.start()

      t1.join()
      t2.join()
      
      return sims1, sims2
    elif len(text) == 3:
      subspace_loc_1 = "/home/kryc1/stories2image/data/subspaces/" + genres[0]
      subspace_loc_2 = "/home/kryc1/stories2image/data/subspaces/" + genres[1]
      subspace_loc_3 = "/home/kryc1/stories2image/data/subspaces/" + genres[2]

      print(subspace_loc_1)
      print(subspace_loc_2)
      print(subspace_loc_3)

      t1 = threading.Thread(target=self.search_compare_thread, args=(subspace_loc_1, sub_txt_total[0], sims1,))
      t2 = threading.Thread(target=self.search_compare_thread, args=(subspace_loc_2, sub_txt_total[1], sims2,))
      t3 = threading.Thread(target=self.search_compare_thread, args=(subspace_loc_3, sub_txt_total[2], sims3,))

      t1.start()
      t2.start()
      t3.start()

      t1.join()
      t2.join()
      t3.join()

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
