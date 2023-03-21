from collections import Counter
from src.common.basic_encoder import BasicEncoder
from src.common.space import Space
from src.text.summarizer import CaptionSummarizer
from src.image.captioning.captioner import Captioner
from src.config import CaptionerConfig, StopWordsConfig


class ImageEncoder(BasicEncoder):

    def __init__(self, space: Space):
        super().__init__(space)
        self.captioner = Captioner(checkpoint_path=CaptionerConfig.checkpoint_path,
                                   word_map_path=CaptionerConfig.word_map_path,
                                   device=CaptionerConfig.device)
        self.summarizer = CaptionSummarizer(stopwords=StopWordsConfig.get_stopwords())

    def create_subspace(self, image_filename: str, dim: int = 5, n_captions: int = 5):
        """Returns a word subspace"""
        _captions: list = self._get_captions(image_filename, n_captions=n_captions)
        _keywords: Counter = self._get_keywords(captions=_captions)
        if _keywords:
            return self.space.create_subspace(keywords=_keywords, dim=dim, use_tf=True)

    def _get_captions(self, image_filename: str, n_captions: int = 5) -> list:
        """ Returns the image captions as a list of str """
        return self.captioner.generate_captions(image_filename=image_filename, n_captions=n_captions)

    def _get_keywords(self, captions: list) -> Counter:
        _keywords = set()
        for caption in captions:
            _keywords.update(self.summarizer.get_keywords(text=caption))
        return Counter(_keywords)


if __name__ == '__main__':
    import sys
    import os
    from gensim.models import KeyedVectors
    from src.config import Word2VecConfig

    sys.path.append(os.path.join(sys.path[0], 'captioning'))  # add models.py

    t_keyed_vectors = KeyedVectors.load_word2vec_format(fname=Word2VecConfig.get_word_vectors_filename(),
                                                        limit=Word2VecConfig.get_vocab_size(),
                                                        binary=True)
    t_space = Space(t_keyed_vectors)
    image_encoder = ImageEncoder(space=t_space)
    t_img = "/root/coco2014/train2014/"
    t_captions: list = image_encoder._get_captions(image_filename=t_img)
    print(t_captions)
    t_keywords: Counter = image_encoder._get_keywords(captions=t_captions)
    print(t_keywords)
