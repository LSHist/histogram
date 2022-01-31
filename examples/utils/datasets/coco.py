import os

from .base import _download, _unpack


class COCOLoader:
    """
    COCO Dataset
    """

    ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"

    def __init__(self, path="datasets/coco"):
        self.coco = None
        self._base_path = path
        self._image_dir = None
        self._annotation_dir = None

    def fetch(self):

        if not os.path.isfile(os.path.join(self._base_path, ".downloaded")):

            # Load the dataset if there doesn't exist the dataset archive
            if not os.path.exists(self._base_path):
                os.makedirs(self._base_path)
            # try:
            #     shutil.rmtree(path)
            # except FileNotFoundError:
            #     os.makedirs(path)

            # Download annotation archive
            archive_path = _download(self.ANNOTATIONS_URL, self._base_path)
            _unpack(archive_path, self._base_path)

            # Download images archive
            archive_path = _download(self.IMAGES_URL, self._base_path)
            _unpack(archive_path, self._base_path)

            # Create the empty file as flag for download status
            with open(os.path.join(self._base_path, ".downloaded"), "w"):
                pass

        self._image_dir = os.path.join(self._base_path, "val2017")
        self._annotation_dir = os.path.join(self._base_path, "annotations", "instances_val2017.json")

        # Open files
        # return {
        #     "annotations_dir": self._annotation_dir,
        #     "images_dir": self._image_dir
        # }

    def load(self):
        from pycocotools.coco import COCO
        self.coco = COCO(annotation_file=os.path.join(self._base_path, "annotations", "instances_val2017.json"))
        return self.coco

    def fetch_load(self):
        self.fetch()
        return self.load()

    def load_path(self, image_id):
        img_meta = self.coco.loadImgs(ids=[image_id])[0]
        return os.path.join(self._image_dir, img_meta["file_name"])

    def load_object_names(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        return [(item["id"], item["name"]) for item in categories]
