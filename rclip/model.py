from typing import Callable, List, Tuple, Optional, cast

import os

import clip
import clip.model
import numpy as np
from PIL import Image
import utils
import torch
import torch.nn
import glob


class Model:
  VECTOR_SIZE = 768
  _device = 'cpu'
  _model_name = 'ViT-L/14@336px'

  def __init__(self, device=None):
    if device is not None:
      self._device = device
    model, preprocess = cast(
      Tuple[clip.model.CLIP, Callable[[Image.Image], torch.Tensor]],
      clip.load(self._model_name, device=self._device)
    )
    self._model = model
    self._preprocess = preprocess

  def compute_image_features(self, images: List[Image.Image]) -> np.ndarray:
    images_preprocessed = torch.stack([self._preprocess(thumb) for thumb in images]).to(self._device)

    images_batches = torch.split(images_preprocessed, 32)
    image_features = None

    for batch in images_batches:
      with torch.no_grad():
        batch_features = self._model.encode_image(batch).cpu()
        image_features = batch_features if image_features is None else torch.cat([image_features, batch_features], dim=0)

    image_features /= image_features.norm(dim=-1, keepdim=True)

    assert image_features.shape[-1] == self.VECTOR_SIZE
    return image_features.cpu().to(torch.float).numpy()

  def compute_text_features(self, text: List[str]) -> np.ndarray:
    with torch.no_grad():
      text_encoded = self._model.encode_text(clip.tokenize(text).to(self._device))
      text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

    assert text_encoded.shape[-1] == self.VECTOR_SIZE
    return text_encoded.to(torch.float).cpu().numpy()

  def group_query_parameters_by_type(self, queries: List[str]) -> Tuple[List[str], List[str], List[str]]:
    phrase_queries: List[str] = []
    local_file_queries: List[str] = []
    url_queries: List[str] = []
    for query in queries:
        if utils.is_http_url(query):
          url_queries.append(query)
        elif utils.is_file_path(query):
          if os.path.isdir(query):
            local_file_queries+=[os.path.join(query, x) for x in os.listdir(query)]
          if "*" not in query:
            local_file_queries.append(query)
          else:
            local_file_queries+=glob.glob(query)
        else:
          phrase_queries.append(query)
    return phrase_queries, local_file_queries, url_queries

  def compute_features_for_queries(self, queries: List[str]) -> np.ndarray:
    text_features: Optional[np.ndarray] = None
    image_features: Optional[np.ndarray] = None
    phrases, files, urls = self.group_query_parameters_by_type(queries)
    if phrases:
      text_features = np.add.reduce(self.compute_text_features(phrases))
    if files or urls:
      images = ([utils.download_image(q) for q in urls] +
                [utils.read_image(q) for q in files])
      image_features = np.mean(self.compute_image_features(images), axis=0)

    if text_features is not None and image_features is not None:
        return text_features + image_features
    elif text_features is not None:
        return text_features
    elif image_features is not None:
        return image_features
    else:
        return np.zeros(Model.VECTOR_SIZE)

  def list_outliers(features): 
    average = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    outliers = []
    for i in range(features):
      if np.abs(features[i]-average)>std:
        outliers.append(i)
    return outliers

  def compute_similarities_to_text(
      self, item_features: np.ndarray,
      positive_queries: List[str], negative_queries: List[str]) -> List[Tuple[float, int]]:

    positive_features = self.compute_features_for_queries(positive_queries)
    negative_features = self.compute_features_for_queries(negative_queries)

    features = positive_features - negative_features

    similarities = features @ item_features.T
    sorted_similarities = sorted(zip(similarities, range(item_features.shape[0])), key=lambda x: x[0], reverse=True)

    return sorted_similarities
