import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image
from typing import List
from IPython.display import Image as ImageOpener, display


class ImageGrouping:
    resnet = models.resnet101(pretrained=True)
    resnet.eval()
    vec_size = 2048
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters

    def get_vector(self, image_name):
        batch_t = torch.unsqueeze(self.transform(Image.open(image_name)), 0)
        v = self.feature_extractor(batch_t)
        v = torch.squeeze(v)
        return v.detach().numpy()

    def group_images(self, image_filenames: list) -> dict:
        count = len(image_filenames)
        vec_mat = np.zeros((count, self.vec_size))
        for i in range(count):
            v = self.get_vector(image_filenames[i])
            vec_mat[i, :] = v
        reduced_data = PCA(n_components=2).fit_transform(vec_mat)
        kmeans = KMeans(init='k-means++', n_clusters=self.n_clusters, n_init=10)
        kmeans.fit(reduced_data)
        preds = kmeans.predict(reduced_data)
        groups = {}
        for i, e in enumerate(preds):
            if groups.get(e) is None:
                groups[e] = [image_filenames[i]]
            else:
                groups[e].append(image_filenames[i])
        return groups

    def get_representatives(self, image_filenames: list) -> list:
        """Returns a list of image filenames"""
        groups: dict = self.group_images(image_filenames=image_filenames)
        return [members[0] for members in groups.values() if members]


def get_random_images(folder: str, count: int = 10) -> List[str]:
    image_filenames = os.listdir(folder)
    n_images = len(image_filenames)
    rand_nums = torch.randint(0, n_images, (count,))
    return [os.path.join(folder, image_filenames[r]) for r in rand_nums]


if __name__ == "__main__":
    from pprint import pprint
    images_folder = "/home/zhanibek/Desktop/Fall '19/Senior Project/news2image/data/images/goi5k"
    img_fns: list = get_random_images(folder=images_folder, count=15)
    ig = ImageGrouping(n_clusters=4)
    # groups: dict = ig.group_images(image_filenames=img_fns)
    reps: list = ig.get_representatives(image_filenames=img_fns)
    pprint(reps)
