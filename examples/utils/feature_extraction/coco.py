import os
import copy
import numpy as np

from .base import create_histogram
from .position import PositionSetTransformer


##################
#
# COCO Dataset
#
##################


def load_coco_histograms(
        coco, limit=5000,
        file_path="histograms/coco.pickle",
        image_splits=(5,5),
        write_to_file=True,
        force_rewrite=False, verbose=False):
    """
    Note: For images
    """

    import time

    histograms_dir = os.path.dirname(file_path)
    if not os.path.exists(histograms_dir):
        os.makedirs(histograms_dir)

    if os.path.isfile(file_path) and not force_rewrite:
        with open(file_path, "rb") as f:
            import pickle
            return pickle.load(f)

    start_tick = time.time()
    hists = list()

    position_transformer = PositionSetTransformer(splits=image_splits, element_ndim=3)
    object_transformer = COCOObjectSetTransformer(coco)

    for indx, (img_id, img_meta) in enumerate(coco.imgs.items()):
        if indx == limit:
            break

        # Note: Just to follow common interface fit_transform
        I = np.zeros((img_meta["height"], img_meta["width"], 3))
        position_image = position_transformer.fit_transform(X=I, y=None)
        object_image = object_transformer.fit_transform(X=None, y=None, ids=img_id)

        # Option 1
        #         merged_image = feature_merger.fit_transform((position_image, object_image))
        #         hist = utils.create_histogram_(merged_image)

        # Option 2
        hist = create_histogram((position_image, object_image))
        hists.append((img_id, hist))
        print("\rCurrent image index: {}/{}".format(indx + 1, limit), end="")

    delta_tick = time.time() - start_tick

    if verbose:
        print("\nTotal time: {}s".format(delta_tick))
        print("Time per image: {}s".format(delta_tick / limit))

    if write_to_file:
        with open(file_path, "wb") as f:
            import pickle
            pickle.dump(hists, f, pickle.HIGHEST_PROTOCOL)

    return hists


"""
Object Transformer
"""


class COCOObjectSetTransformer:

    def __init__(self, coco):
        self._coco = coco
        categories = coco.loadCats(coco.getCatIds())
        self._ids_to_names = {str(item["id"]): item["name"] for item in categories}

    def fit(self, X, y=None):
        return self

    def transform(self, X, batch=False, ids=None):
        if isinstance(ids, int) and batch is False:
            object_regions = self._load_image_regions_coco(ids)
            image_meta = self._coco.loadImgs(ids=[ids])[0]
            shape = (image_meta["height"], image_meta["width"])
            return self._create_object_mask(object_regions, shape)
        # elif isinstance(ids, (np.ndarray, set, list, tuple)) and batch is True:
        #     return self.data["labels"][np.array(tuple(ids))]
        # elif batch is True and ids is None:
        #     return self.data["labels"]
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

    def get_element_name(self, element_ids, return_id=True):
        result = list()
        if isinstance(element_ids, (set, list, tuple, np.ndarray)):
            for element_id in element_ids:
                if str(element_id) in self._ids_to_names:
                    if return_id is True:
                        result.append((self._ids_to_names[str(element_id)], element_id))
                    else:
                        result.append(self._ids_to_names[str(element_id)])
            return result
        else:
            if str(element_ids) in self._ids_to_names:
                if return_id is True:
                    return self._ids_to_names[str(element_ids)]
                else:
                    return self._ids_to_names[str(element_ids)], element_ids

    def _create_object_mask(self, object_regions, shape):
        import skimage.draw as draw
        obj_mask = np.full(shape, fill_value=0, dtype=np.int)
        self._element_regions = copy.deepcopy(object_regions)
        self._element_mapping = dict()
        for i in range(len(object_regions)):
            object_id = str(object_regions[i][0])
            if object_id not in self._element_mapping:
                self._element_mapping[object_id] = list()
            self._element_mapping[object_id].append(i)
            r, c = draw.polygon(object_regions[i][1][:, 1], object_regions[i][1][:, 0])
            obj_mask[r, c] = object_regions[i][0]
        return obj_mask

    def _load_image_regions_coco(self, image_id):
        image_anns_id = self._coco.getAnnIds(imgIds=image_id, iscrowd=False)
        image_anns = self._coco.loadAnns(image_anns_id)
        image_regions = list()
        for i in range(len(image_anns)):
            if image_anns[i]["iscrowd"] == 0:
                seg_ = image_anns[i]["segmentation"][0]
                poly_ = np.array(seg_).reshape((len(seg_) // 2, 2))
                image_regions.append((image_anns[i]["category_id"], poly_))
        return image_regions



