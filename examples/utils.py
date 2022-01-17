import os
import shutil
import urllib.request
import numpy as np
import itertools
from pycocotools.coco import COCO

from himpy.executor import Parser, Evaluator
from himpy.histogram import Histogram, Histogram1D, HElement, HElementSet


def fetch_coco(path="datasets/coco"):

    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    images_url = "http://images.cocodataset.org/zips/val2017.zip"

    if not os.path.isfile(os.path.join(path, ".downloaded")):

        # Load the dataset if there doesn't exist the dataset archive
        if not os.path.exists(path):
            os.makedirs(path)
        # try:
        #     shutil.rmtree(path)
        # except FileNotFoundError:
        #     os.makedirs(path)

        # Download annotation archive
        _download(annotations_url, path)
        
        # Download images archive
        _download(images_url, path)

        # Create the empty file as flag for download status
        with open(os.path.join(path, ".downloaded"), "w"):
            pass

    # Open files
    return {
        "annotations_dir": os.path.join(path, "annotations", "instances_val2017.json"),
        "images_dir": os.path.join(path, "val2017")
    }


def load_image_path_coco(coco, image_id, images_dir):
    img_meta = coco.loadImgs(ids=[image_id])[0]
    return os.path.join(images_dir, img_meta["file_name"])


def load_object_names_coco(coco):
    categories = coco.loadCats(coco.getCatIds())
    return [(item["id"], item["name"]) for item in categories]


def load_image_regions_coco(coco, image_id):
    image_anns_id = coco.getAnnIds(imgIds=image_id, iscrowd=False)
    image_anns = coco.loadAnns(image_anns_id)
    image_regions = list()
    for i in range(len(image_anns)):
        if image_anns[i]["iscrowd"] == 0:
            seg_ = image_anns[i]["segmentation"][0]
            poly_ = np.array(seg_).reshape((len(seg_) // 2, 2))
            image_regions.append((image_anns[i]["category_id"], poly_))
    return image_regions


def _download(url, path):

    def show_download_progress(num, size, total):
        num_total = total / size
        progress = int(num / num_total * 100)
        print("\r[{:100}]".format("=" * progress + (">" if progress != 100 else "")), end="")

    filename = os.path.basename(url)
    archive_path = os.path.join(path, filename)
    tmp_filename = "~" + filename
    tmp_archive_path = os.path.join(path, tmp_filename)
    print("Downloading {}...".format(filename))
    urllib.request.urlretrieve(url, tmp_archive_path, reporthook=show_download_progress)
    print("\nCompleted.")
    os.rename(tmp_archive_path, archive_path)
    shutil.unpack_archive(archive_path, path)
    os.unlink(archive_path)


def load_coco_histograms(
        coco, limit=5000, file_path="imagehist.pickle",
        image_splits=(5,5),
        write_to_file=True,
        force_rewrite=False, verbose=False):
    """
    Note: For images
    """

    import time
    if os.path.isfile(file_path) and not force_rewrite:
        with open(file_path, "rb") as f:
            import pickle
            return pickle.load(f)

    start_tick = time.time()
    hists = list()

    position_transformer = PositionSetTransformer(splits=image_splits)
    object_transformer = ObjectSetTransformer(load_object_names_coco(coco))

    for indx, (img_id, img_meta) in enumerate(coco.imgs.items()):
        if indx == limit:
            break
        image_regions = load_image_regions_coco(coco, img_id)
        position_image = position_transformer.fit_transform(shape=(img_meta["height"], img_meta["width"]))
        object_image = object_transformer.transform(image_regions, shape=(img_meta["height"], img_meta["width"]))

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


# TODO: Move to utils module
class SearchEngine:

    def __init__(self, hists, parser: Parser, evaluator: Evaluator):
        self._hists = hists
        self._parser = parser
        self._evaluator = evaluator

    def retrieve(self, query, topN=10, lastN=None, threshold=0.001):
        img_rank = list()
        if hasattr(query, "value") and isinstance(query.value, str):
            """Searching by expression"""
            expr = self._parser.parse_string(query.value)
            HEs = [(img_id, self._evaluator.eval(expr, hist)) for img_id, hist in self._hists]
            img_rank = sorted(
                [(img_id, HE.sum()) for img_id, HE in HEs if HE.sum() > threshold],
                key=lambda x: -x[1]
            )
        elif isinstance(query, Histogram):
            """Searching by data histogram"""
            img_rank = sorted(
                [(image_id, (query * hist).sum()) for image_id, hist in self._hists],
                key=lambda x: -x[1]
            )
        if isinstance(lastN, int):
            return img_rank[:topN], img_rank[-lastN:]

        return img_rank[:topN]


# def create_histogram_(features):
#     features_flatten = tuple([item.flatten() for item in features])
#     features_ = np.c_[features_flatten]
#     hist = Histogram1D(data=None)
#     for item in features_:
#         item_tuple = tuple(map(str, item.tolist()))
#         if item_tuple not in hist:
#             hist[item_tuple] = HElement(item_tuple, 0)
#         hist[item_tuple].value += 1
#     hist.normalize(features_.shape[0])
#     return hist


# TODO: Combine create_histogram and create_histogram_
def create_histogram(features, normalize=True):
    """
    Faster than create_histogram_
    """
    features_ = None
    ndim_features = len(features)
    features_flatten = tuple([item.reshape(-1) for item in features])
    if ndim_features == 1:
        features_ = features_flatten[0]
    elif ndim_features > 1:
        # None: if some feature is str, then all features_ will be str
        features_ = np.c_[features_flatten]
        # preserve only items with non-zero element
        # Note: zero element means NaN as element ids is not equal to 0
        zero_ = 0 if np.issubdtype(features_.dtype, np.integer) else "0"
        features_ = features_[np.all(features_ != zero_, axis=1)]
        # for i in range(features_.shape[1]):
        #     features_ = features_[features_[:,i]!=0]

    num_elements = features_flatten[0].shape[0]
    elements, counts = np.unique(features_, axis=0, return_counts=True)
    if normalize:
        counts = counts / num_elements
    hist = Histogram1D(data=None)
    for i in range(len(counts)):
        item_tuple = elements[i]
        if ndim_features > 1:
            item_tuple = tuple(map(str, elements[i].tolist()))
        hist[item_tuple] = HElement(item_tuple, counts[i])
    return hist


def create_histogram_(merged_features, normalize=True):
    """
    Slower than create_histogram?
    """
    features_ = merged_features.reshape(-1)
    ndim_element = len(features_[0]) if isinstance(features_[0], np.record) else 1

    if ndim_element > 1:
        # preserve only items with non-zero element
        # Note: zero element means NaN as element ids is not equal to 0
        for name in features_.dtype.names:
            # check type str or int to use appropriate zero
            zero_ = 0 if np.issubdtype(features_[name].dtype, np.integer) else "0"
            features_ = features_[features_[name] != zero_]

    elements, counts = np.unique(features_, axis=0, return_counts=True)
    if normalize:
        counts = counts / merged_features.size
    hist = Histogram1D(data=None)
    for i in range(len(counts)):
        item_tuple = elements[i]
        if ndim_element > 1:
            item_tuple = tuple(map(str, elements[i].tolist()))
        hist[item_tuple] = HElement(item_tuple, counts[i])
    return hist


def merge_features(*features):
    features_ = None
    shape = features[0].shape
    ndim_features = len(features)
    features_flatten = tuple([item.reshape(-1) for item in features])
    if ndim_features == 1:
        features_ = np.rec.fromarrays(features_flatten[0])
    elif ndim_features > 1:
        features_ = np.rec.fromarrays(features_flatten)
    return features_.reshape(shape)


def filter_data(data, features, elements):

    # TODO: check?
    # def to_int(items):
    #     return map(int, items)

    features_ = features.reshape(-1)
    if len(data.shape) == len(features.shape) + 1:
        data_flatten = data.reshape(-1, data.shape[-1])
    elif len(data.shape) == len(features.shape):
        data_flatten = data.reshape(-1)
    else:
        raise Exception()

    mask = np.full(data_flatten.shape, fill_value=255, dtype=np.int)

    # features_ = np.rec.fromarrays(features_flatten)
    # elements_ = np.array(
    #     list(itertools.product(*(to_int(items) for items in elements))),
    #     dtype=features_.dtype)

    # elements_ = np.array(
    #     list(itertools.product(*(items for items in elements))),
    #     dtype=features_.dtype)

    elements_ = np.array(elements, dtype=features_.dtype)

    indx = np.in1d(features_, elements_)
    mask[indx] = data_flatten[indx]

    return mask.reshape(data.shape)


def extract_element_set(HE, none_dim=1):
    """

    Parameters
    ----------
    HE : HElementSet
    none_dim : int
        used to return empty dicts if no elements

    Returns
    -------

    """
    elements = list(HE.to_dict().keys())
    if len(elements) == 0 and none_dim > 1:
        return [dict() for _ in range(none_dim)]
    elif len(elements) == 0:
        return None
    dim = len(elements[0])
    result = list()
    for i in range(dim):
        result.append({el[i] for el in elements})
    return result


class FeatureMerger:
    def __init__(self):
        self._ndim = None

    def fit(self, X, single=True):
        self._ndim = len(X) if single else len(X[0])
        return self

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def transform(self, X, single=True):
        if self._ndim is None:
            raise Exception("Use the fit method at first.")
        ndim = len(X) if single else len(X[0])
        if self._ndim != ndim:
            raise Exception("Mismatching dimensions.")
        if single:
            features_ = None
            shape = X[0].shape
            features_flatten = tuple([item.reshape(-1) for item in X])
            if ndim == 1:
                features_ = features_flatten[0]
            elif ndim > 1:
                features_ = np.rec.fromarrays(features_flatten)
            return features_.reshape(shape)
        else:
            raise NotImplementedError


class ObjectSetTransformer:

    def __init__(self, element_names):
        self._element_names = {str(element_id): element_name for element_id, element_name in element_names}
        self._element_regions = None
        self._element_mapping = None
        self._size = None
        # self._data_meta = None
        pass

    def fit(self):
        pass

    def fit_transform(self):
        pass

    def transform(self, object_regions, shape):
        import skimage.draw as draw
        import copy
        self._size = shape
        obj_mask = np.full(self._size, fill_value=0, dtype=np.int)
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

    def filter_data(self, data, element_ids=None):
        import skimage.draw as draw
        mask = np.full(data.shape, fill_value=0, dtype=np.int)
        if isinstance(element_ids, (str, int)):
            region_ = self.find_element(element_ids)
            r, c = draw.polygon(region_[:, 1], region_[:, 0])
            mask[r, c, :] = data[r, c, :]
            return mask
        if isinstance(element_ids, (tuple, list, set)):
            for element_id in element_ids:
                regions_ = self.find_element(element_id)
                for region_ in regions_:
                    r, c = draw.polygon(region_[:, 1], region_[:, 0])
                    mask[r, c, :] = data[r, c, :]
            return mask
        if element_ids is None:
            for element_id, _ in self._element_mapping.items():
                regions_ = self.find_element(element_id)
                for region_ in regions_:
                    r, c = draw.polygon(region_[:, 1], region_[:, 0])
                    mask[r, c, :] = data[r, c, :]
            return mask

    def filter_elements(self, element_ids):
        import skimage.draw as draw
        pos_mask = np.zeros(self._size, dtype=np.int)
        if isinstance(element_ids, (str, int)):
            regions_ = self.find_element(element_ids)
            for region_ in regions_:
                r, c = draw.polygon(region_[:, 1], region_[:, 0])
                pos_mask[r, c] = element_ids
            return pos_mask
        if isinstance(element_ids, (tuple, list, set)):
            for element_id in element_ids:
                regions_ = self.find_element(element_id)
                for region_ in regions_:
                    r, c = draw.polygon(region_[:, 1], region_[:, 0])
                    pos_mask[r, c] = element_id
            return pos_mask

    def find_element(self, element_id):
        if element_id is not None and element_id != "":
            if element_id in self._element_mapping:
                for region_id in self._element_mapping[element_id]:
                    yield self._element_regions[region_id][1]
        return None

    def find_elements(self, element_ids):
        if element_ids is not None and element_ids != "":
            for element_id in element_ids:
                for region_id in self._element_mapping[element_id]:
                    yield self._element_regions[region_id]
        else:
            for element in self:
                yield element

    def find_element_name(self, element_id):
        return self._element_names.get(element_id)

    def find_element_names(self, element_ids):
        names = list()
        for element_id in element_ids:
            names.append(self._element_names.get(element_id))
        return names

    def __iter__(self):
        for element_id, element_region in self._element_regions:
            yield element_id, element_region


class PositionSetTransformer:

    def __init__(self, splits):
        if not isinstance(splits, int) and not isinstance(splits, (list, tuple)):
            raise Exception("Provided splits type is not supported.")
        self._splits = splits
        if isinstance(splits, int) :
            self._dim = 1
        elif isinstance(splits, (list, tuple)):
            self._dim = len(splits)
        self._steps = None
        self._size = None
        self._element_regions = None
        self._element_mapping = None
        self._last_transformed = None

    def fit(self, data=None, shape=None):
        """
        Note:
            Only numpy array supported as X
        """
        # TODO: Is self._compose_low_elements() used only in the fit method?

        if data is None and shape is None:
            raise Exception()
        if shape is None:
            size = data.shape
        else:
            size = shape

        if isinstance(shape, int) and self._dim != 1 or \
                isinstance(shape, (tuple, list, set)) and len(size) != self._dim:
            raise Exception("Mismatched data dimensions.")
        # Avoid recalculation elements if a previous fit call had the same size
        if self._size is None or size != self._size:
            self._size = size
            self._last_transformed = None
            self._compose_low_elements()

        return self

    def fit_transform(self, data=None, shape=None):
        return self.fit(data, shape).transform()

    def transform(self):
        if self._last_transformed is not None:
            return self._last_transformed
        return self._transform()

    def transform_all(self, data):
        raise NotImplementedError()

    # TODO: make return of find_element and find_elements in same manner
    def find_element(self, element_id):
        if element_id is not None and element_id != "":
            if element_id in self._element_mapping:
                return self._element_regions[self._element_mapping[element_id]]
        return None

    def find_elements(self, element_ids):
        if element_ids is not None and element_ids != "":
            for element_id in element_ids:
                if element_id in self._element_mapping:
                    yield element_id, self._element_regions[self._element_mapping[element_id]]
        else:
            for element in self:
                yield element

    def filter_elements(self, element_ids):
        pos_mask = np.zeros(self._size, dtype=np.int)
        if isinstance(element_ids, (str, int)):
            element_ = self.find_element(element_ids)
            for low_element in itertools.product(
                    *[range(element_[i], element_[self._dim + i] + 1) for i in range(self._dim)]
            ):
                pos_mask[low_element] = element_ids
            return pos_mask

        if isinstance(element_ids, (tuple, list, set)):
            for element_id in element_ids:
                element_ = self.find_element(element_id)
                for low_element in itertools.product(
                    *[range(element_[i], element_[self._dim+i]+1) for i in range(self._dim)]
                ):
                    pos_mask[low_element] = element_id
            return pos_mask

    def filter_data(self, data, element_ids):
        mask = np.full(data.shape, fill_value=0, dtype=np.int)
        if isinstance(element_ids, (str, int)):
            element_region = self.find_element(element_ids)
            for low_element in self._mask_element(element_region):
                mask[low_element] = data[low_element]
            return mask
        if isinstance(element_ids, (tuple, list, set)):
            for element_id in element_ids:
                element_region = self.find_element(element_id)
                for low_element in self._mask_element(element_region):
                    mask[low_element] = data[low_element]
            return mask

    def _transform(self):
        mask = np.zeros(self._size, dtype=np.int)
        for element_id, element_region in self:
            for low_element in itertools.product(
                *[range(element_region[i], element_region[self._dim+i]+1) for i in range(self._dim)]
            ):
                mask[low_element] = element_id
        self._last_transformed = mask
        return mask

    def _compose_low_elements(self):
        if isinstance(self._size, int):
            step = self._size//self._splits
            abs_intervals = self._build_absolute_intervals(step, self._splits)
            self._element_regions = list()
            for start in abs_intervals:
                self._element_regions.append((start, start + step - 1))
            self._build_position_elements()
        elif isinstance(self._size, (list, tuple)):
            steps = list()
            abs_intervals = list()
            for i, dim_size in enumerate(self._size):
                steps.append(dim_size // self._splits[i])
                abs_intervals.append(self._build_absolute_intervals(steps[i], self._splits[i]))
            self._element_regions = list()
            for start in itertools.product(*abs_intervals):
                grid_start = list()
                grid_end = list()
                for i, start_dim_item in enumerate(start):
                    grid_start.append(start_dim_item)
                    grid_end.append(start_dim_item + steps[i] - 1)
                self._element_regions.append(tuple(grid_start + grid_end))
            self._build_position_elements()
        else:
            raise Exception("Provided unsupported type.")

    def _build_absolute_intervals(self, dim_step, dim_splits):
        return [i*dim_step for i in range(dim_splits)]

    def _build_position_elements(self):
        self._element_mapping = dict()
        for i in range(len(self._element_regions)):
            self._element_mapping[str(i + 1)] = i

    def _mask_element(self, element_region):
        for low_elements in itertools.product(
            *[range(element_region[i], element_region[self._dim+i]+1) for i in range(self._dim)]
        ):
            yield low_elements

    def __iter__(self):
        for key, value in iter(self._element_mapping.items()):
            yield key, self._element_regions[value]


def plot_position_grid(position_transformer, ax, element_ids=None):
    from matplotlib.patches import Rectangle
    for el in position_transformer.find_elements(element_ids):
        bottom, height = el[1][0], el[1][2]-el[1][0]
        left, width = el[1][1], el[1][3]-el[1][1]
        right = left + width
        top = bottom + height
        ax.add_patch(Rectangle(xy=(left, bottom), width=width, height=height, fill=False,
                                      label=el, edgecolor="red", linewidth=2))
        ax.text(0.5*(left+right), 0.5*(bottom+top), el[0],
                horizontalalignment="center", verticalalignment="center", fontsize=15, color="red")
    return ax


def plot_object_edges(object_transformer, ax, element_ids=None):
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    seg_polys = list()
    for element_id, element_region in object_transformer.find_elements(element_ids):
        center = element_region.mean(axis=0)
        ax.text(*center, str(element_id),
                horizontalalignment="center", verticalalignment="center", fontsize=15, color="yellow")
        seg_polys.append(Polygon(element_region, fill=False, label="text"))
    p_objs = PatchCollection(seg_polys, edgecolor="black", alpha=0.2, linewidths=2)
    ax.add_collection(p_objs)
    return ax


def show_operation_result(I, merged_image, HE1, HE2, HE3, transformers, titles=("E1", "E2", "Result")):
    import matplotlib.pyplot as plt

    E1_set = extract_element_set(HE1, 2)
    E2_set = extract_element_set(HE2, 2)
    E3_set = extract_element_set(HE3, 2)

    E1_image = filter_data(I, merged_image, HE1.elements())
    E2_image = filter_data(I, merged_image, HE2.elements())
    E3_image = filter_data(I, merged_image, HE3.elements())

    fig, axes = plt.subplots(1, 3, figsize=(14, 20))
    axes[0].set_title(titles[0])
    axes[0].imshow(E1_image)
    axes[0] = plot_position_grid(transformers[0], axes[0], E1_set[0])
    axes[0] = plot_object_edges(transformers[1], axes[0], E1_set[1])
    axes[0].get_xaxis().set_visible(False)
    axes[0].get_yaxis().set_visible(False)
    axes[1].set_title(titles[1])
    axes[1].imshow(E2_image)
    axes[1] = plot_position_grid(transformers[0], axes[1], E2_set[0])
    axes[1] = plot_object_edges(transformers[1], axes[1], E2_set[1])
    axes[1].get_xaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)
    axes[2].set_title(titles[2])
    axes[2].imshow(E3_image)
    axes[2] = plot_position_grid(transformers[0], axes[2], E3_set[0])
    axes[2] = plot_object_edges(transformers[1], axes[2], E3_set[1])
    axes[2].get_xaxis().set_visible(False)
    axes[2].get_yaxis().set_visible(False)

    plt.show()


def show_retrieved_images(ranked_images, image_paths, title=None, limit=11, cols=5):
    import matplotlib.pyplot as plt
    import matplotlib.image as image_utils

    img_limit = len(ranked_images) if limit > len(ranked_images) else limit
    #     if limit and img_limit > limit:
    #         img_limit = limit
    row_num = -(-img_limit // cols)

    # show title
    fig, ax = plt.subplots(1, figsize=(15, 1))
    ax.text(0.5, 0.5, title, clip_on=True, ha="center", va="center", fontsize=16, weight="bold")
    ax.axis("off")
    plt.show()

    # show images
    fig, axs = plt.subplots(row_num, cols, figsize=(15, 4 * row_num), squeeze=False)
    for i in range(row_num):
        for j in range(cols):
            indx = i * cols + j
            if indx >= img_limit:
                fig.delaxes(axs[i, j])
            else:
                I = image_utils.imread(image_paths[indx])
                axs[i, j].imshow(I)
                axs[i, j].set_title(
                    "rank={}\nid={}\nscore={:0.4f}".format(
                        indx + 1,
                        ranked_images[indx][0],
                        ranked_images[indx][1]))
                axs[i, j].axis("off")
    plt.tight_layout()
    plt.show()


def convert_complete_histogram_values(U, H, to_sort=False):
    """
    Convert a sparse form of a histogram to a list of values of all elements.
    Absent elements in the sparse form will be replaced by zero values.

    Parameters
    ----------
    U - the universal set of elements
    H - histogram of data or element
    to_sort - whether it's needed to sort elements by names

    Returns
    -------
    Values of all elements
    """
    hist_val_full = [0 for _ in U]

    if isinstance(H, Histogram):
        for i in range(len(U)):
            if U[i] in H:
                hist_val_full[i] = H[U[i]].value
    elif isinstance(H, HElementSet):
        elements = H.to_dict()
        for i in range(len(U)):
            if U[i] in elements:
                hist_val_full[i] = elements[U[i]]

    if to_sort:
        hist_val_full.sort(reverse=False)

    return hist_val_full


if __name__ == "__main__":
    pass
