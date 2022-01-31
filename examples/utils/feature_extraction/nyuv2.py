import numpy as np

from .base import FeatureMerger, create_histogram_batch_
from .position import PositionSetTransformer


##################
#
# NYU Depth V2 Dataset
#
##################


def load_nyu_histograms(
        data,
        batch_size=100,
        file_path="histograms/nyuv2hist.pickle",
        write_to_file=True,
        force_rewrite=False,
        verbose=True):
    import os, time

    histograms_dir = os.path.dirname(file_path)
    if not os.path.exists(histograms_dir):
        os.makedirs(histograms_dir)

    if os.path.isfile(file_path) and not force_rewrite:
        with open(file_path, "rb") as f:
            import pickle
            return pickle.load(f)

    start_tick = time.time()

    # Create feature transformers

    position_transformer = PositionSetTransformer(splits=(5, 5), element_ndim=3)
    depth_transformer = NYUDepthSetTransformer(data, nbins=10)
    object_transformer = NYUObjectSetTransformer(data)

    feature_merger = FeatureMerger()

    # Note: X and y are None because we don't actually extract features,
    #  instead we use predefined dataset. X and y are included to follow
    #  the basic interface.

    count = len(data["images"])
    num_batches, last_batch_size = divmod(count, batch_size)

    hists = list()

    print("\rNumber of processed images: 0", end="")

    for i in range(num_batches + 1):
        ids = list(range(i * batch_size, (i + 1) * batch_size))
        if i == num_batches:
            ids = list(range(num_batches * batch_size, num_batches * batch_size + last_batch_size))
        position_images = np.array(
            position_transformer.fit_transform(X=data["images"], y=None, ids=ids, batch=True))
        depth_images = depth_transformer.fit_transform(X=data["images"], y=None, ids=ids, batch=True)
        object_images = object_transformer.fit_transform(X=data["images"], y=None, ids=ids, batch=True)
        merged_images = feature_merger.fit_transform((position_images, depth_images, object_images))
        hists.extend(
            zip(ids, create_histogram_batch_(merged_images))
        )
        print("\rNumber of processed images: {}".format(len(hists)), end="")

    delta_tick = time.time() - start_tick

    if verbose:
        print("\nTotal time: {}s".format(delta_tick))
        print("Time per image: {}s".format(delta_tick / len(hists)))

    if write_to_file is True:
        with open(file_path, "wb") as f:
            import pickle
            pickle.dump(hists, f, pickle.HIGHEST_PROTOCOL)

    return hists



"""
Object Transformer
"""


class NYUObjectSetTransformer:

    def __init__(self, data):
        self.data = data

    def fit(self, X, y=None):
        return self

    def transform(self, X, batch=False, ids=None):
        """
        Transform images to object masks.

        Parameters
        ----------
        X : array
            Input array that contains images. In this case this argument is useless as we use already segmented images.
            So, instead of using X we use the ids argument to specify images to get their segmented objects
        single : bool
            Whether we pass single image, True, or multiple as array, False
        ids : set, list, tuple
            Ids of images from dataset

        Returns
        -------
        array
            object masks of images
        """

        if isinstance(ids, int) and batch is False:
            return self.data["labels"][ids]
        elif isinstance(ids, (np.ndarray, set, list, tuple)) and batch is True:
            return self.data["labels"][np.array(tuple(ids))]
        elif batch is True and ids is None:
            return self.data["labels"]
        else:
            raise ValueError

    def fit_transform(self, X, y=None, batch=False, ids=None):
        return self.fit(X, y).transform(X, batch=batch, ids=ids)

    def filter_data(self, X, I, element_ids=None):
        if element_ids and not isinstance(element_ids, (set, list, tuple)):
            raise ValueError
        if len(X.shape) == 2:
            mask = np.full(I.shape, fill_value=255, dtype=np.uint8)
            if element_ids is None:
                indices = np.nonzero(X)
                mask[indices] = I[indices]
            else:
                filtering_elements = np.array(tuple(element_ids), dtype=np.uint8)
                indices = np.nonzero(np.isin(X, filtering_elements))
                mask[indices] = I[indices]
            return mask

    def filter_elements(self, X, element_ids=None):
        if not isinstance(element_ids, (set, list, tuple)):
            raise ValueError
        if element_ids is None:
            return X
        if len(X.shape) == 2:
            mask = np.zeros(X.shape, dtype=np.uint8)
            filtering_elements = np.array(tuple(element_ids), dtype=np.uint8)
            indices = np.nonzero(np.isin(X, filtering_elements))
            mask[indices] = X[indices]
            return mask

    def get_names(self, element_ids, return_id=True):
        result = list()
        if isinstance(element_ids, (set, list, tuple, np.ndarray)):
            for element_id in element_ids:
                if str(element_id) in self.data["idsToNames"]:
                    if return_id is True:
                        result.append((self.data["idsToNames"][str(element_id)], element_id))
                    else:
                        result.append(self.data["idsToNames"][str(element_id)])
            return result
        else:
            if str(element_ids) in self.data["idsToNames"]:
                if return_id is True:
                    return self.data["idsToNames"][str(element_ids)]
                else:
                    return self.data["idsToNames"][str(element_ids)], element_ids


"""
Depth Transformer
"""


class NYUDepthSetTransformer:

    def __init__(self, data, nbins=10):
        self.data = data
        self.nbins = nbins
        self._bins = None

    def fit(self, X, y=None):
        min_depth, max_depth = self.data["depths"].min(), self.data["depths"].max()
        self._bins = np.linspace(min_depth, max_depth, self.nbins)
        return self

    def transform(self, X, batch=False, ids=None):
        if isinstance(ids, int) and batch is False:
            return np.digitize(self.data["depths"][ids], bins=self._bins)
        elif isinstance(ids, (np.ndarray, set, list, tuple)) and batch is True:
            return np.digitize(self.data["depths"][np.array(tuple(ids))], bins=self._bins)
        elif batch is True and ids is None:
            return np.digitize(self.data["depths"], bins=self._bins)
        else:
            raise ValueError

    def fit_transform(self, X, y=None, batch=False, ids=None):
        return self.fit(X, y).transform(X, batch=batch, ids=ids)

    def filter_data(self, X, I, element_ids=None):
        if element_ids and not isinstance(element_ids, (set, list, tuple)):
            raise ValueError
        if len(X.shape) == 2:
            mask = np.full(I.shape, fill_value=255, dtype=np.uint8)
            if element_ids is None:
                indices = np.nonzero(X)
                mask[indices] = I[indices]
            else:
                filtering_elements = np.array(tuple(element_ids), dtype=np.uint8)
                indices = np.nonzero(np.isin(X, filtering_elements))
                mask[indices] = I[indices]
            return mask

    def filter_elements(self, X, element_ids=None):
        if not isinstance(element_ids, (set, list, tuple)):
            raise ValueError
        if element_ids is None:
            return X
        if len(X.shape) == 2:
            mask = np.zeros(X.shape, dtype=np.uint8)
            filtering_elements = np.array(tuple(element_ids), dtype=np.uint8)
            indices = np.nonzero(np.isin(X, filtering_elements))
            mask[indices] = X[indices]
            return mask
