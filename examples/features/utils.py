import colorsys
import json
import sys
import warnings
import random
import math
import copy

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from plotly.subplots import make_subplots


sys.path.insert(0, "../../")
from himpy.histogram import Histogram, HElementSet, Histogram1D, HElement
from himpy.utils import E


COLOR_ELEMENTS = (
    {"id": "e1", "h": [40, 54], "s": [30, 79], "b": [20, 59]},
    {"id": "e2", "h": [40, 54], "s": [80, 159], "b": [20, 59]},
    {"id": "e3", "h": [40, 54], "s": [160, 240], "b": [20, 59]},
    {"id": "e4", "h": [55, 104], "s": [0, 29], "b": [60, 159]},
    {"id": "e5", "h": [55, 104], "s": [30, 79], "b": [20, 59]},
    {"id": "e6", "h": [55, 104], "s": [30, 79], "b": [60, 159]},
    {"id": "e7", "h": [55, 104], "s": [30, 79], "b": [160, 219]},
    {"id": "e8", "h": [55, 104], "s": [30, 79], "b": [220, 240]},
    {"id": "e9", "h": [55, 104], "s": [80, 159], "b": [20, 59]},
    {"id": "e10", "h": [55, 104], "s": [80, 159], "b": [60, 159]},
    {"id": "e11", "h": [55, 104], "s": [80, 159], "b": [160, 219]},
    {"id": "e12", "h": [55, 104], "s": [80, 159], "b": [220, 240]},
    {"id": "e13", "h": [55, 104], "s": [160, 240], "b": [20, 59]},
    {"id": "e14", "h": [55, 104], "s": [160, 240], "b": [60, 159]},
    {"id": "e15", "h": [55, 104], "s": [160, 240], "b": [160, 219]},
    {"id": "e16", "h": [55, 104], "s": [160, 240], "b": [220, 240]},
    {"id": "e17", "h": [105, 114], "s": [30, 79], "b": [60, 159]},
    {"id": "e18", "h": [105, 114], "s": [80, 159], "b": [60, 159]},
    {"id": "e19", "h": [105, 114], "s": [160, 240], "b": [20, 59]},
    {"id": "e20", "h": [105, 114], "s": [160, 240], "b": [60, 159]},
    {"id": "e21", "h": [30, 39], "s": [30, 79], "b": [220, 240]},
    {"id": "e22", "h": [40, 54], "s": [30, 79], "b": [60, 159]},
    {"id": "e23", "h": [40, 54], "s": [30, 79], "b": [160, 219]},
    {"id": "e24", "h": [40, 54], "s": [30, 79], "b": [220, 240]},
    {"id": "e25", "h": [40, 54], "s": [80, 159], "b": [60, 159]},
    {"id": "e26", "h": [40, 54], "s": [80, 159], "b": [160, 219]},
    {"id": "e27", "h": [40, 54], "s": [80, 159], "b": [220, 240]},
    {"id": "e28", "h": [43, 54], "s": [160, 240], "b": [60, 159]},
    {"id": "e29", "h": [40, 54], "s": [160, 240], "b": [220, 240]},
    {"id": "e30", "h": [45, 54], "s": [160, 240], "b": [160, 219]},
    {"id": "e31", "h": [225, 9], "s": [30, 79], "b": [60, 159]},
    {"id": "e32", "h": [225, 9], "s": [30, 79], "b": [160, 219]},
    {"id": "e33", "h": [225, 9], "s": [80, 159], "b": [20, 59]},
    {"id": "e34", "h": [225, 9], "s": [80, 159], "b": [60, 159]},
    {"id": "e35", "h": [225, 9], "s": [80, 159], "b": [160, 219]},
    {"id": "e36", "h": [225, 9], "s": [80, 159], "b": [220, 240]},
    {"id": "e37", "h": [225, 9], "s": [160, 240], "b": [20, 59]},
    {"id": "e38", "h": [225, 9], "s": [160, 240], "b": [60, 159]},
    {"id": "e39", "h": [225, 9], "s": [160, 240], "b": [160, 219]},
    {"id": "e40", "h": [225, 9], "s": [160, 240], "b": [220, 240]}
)


def rgb2hsl(r, g, b):
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return h * 240, s * 240, l * 240


def hsl2rgb(h, s, b):
    r, g, b = colorsys.hls_to_rgb(h/240, b/240, s/240)
    return round(255 * r), round(255 * g), round(255 * b)


def get_h_avg(h):
    return 0.0 if h[0] > h[1] else h[0] + (h[1] - h[0]) / 2


COLOR_ELEMENTS_RGB = {
    el["id"]: hsl2rgb(
        get_h_avg(el["h"]),
        el["s"][0] + (el["s"][1] - el["s"][0]) / 2,
        el["b"][0] + (el["b"][1] - el["b"][0]) / 2)
    for el in COLOR_ELEMENTS
}


class ColorImageGenerator:

    def __init__(self, elements=None, element_type="hsb40"):
        self._elements = None
        if element_type == "hsb40":
            self._elements = copy.deepcopy(COLOR_ELEMENTS_RGB)
        elif elements is not None:
            self._elements = copy.deepcopy(elements)
        else:
            raise AttributeError

    def generate(self, shape, element_ids="all", steps=(5, 5), normal_element_ids=None, random_state=None):

        random.seed(random_state)

        num_splits_x = shape[1] // steps[1]
        num_splits_y = shape[0] // steps[0]

        element_ids_ = list(self._elements.keys()) if element_ids == "all" else element_ids

        image = np.zeros(list(shape) + [3], dtype=np.uint8)

        for i in range(num_splits_y):
            for j in range(num_splits_x):
                image[i*steps[1]:(i+1)*steps[1], j*steps[0]:(j+1)*steps[0], :] = \
                    self._elements[random.choice(list(element_ids_))]

        if not normal_element_ids:
            return image

        cx = random.randint(0, num_splits_x)
        cy = random.randint(0, num_splits_y)

        sx = num_splits_x//4
        sy = num_splits_y//4

        for _ in range(num_splits_y):
            for _ in range(num_splits_x):
                x = round(random.normalvariate(mu=cx, sigma=sx))
                y = round(random.normalvariate(mu=cy, sigma=sy))
                image[x*steps[1]:(x+1)*steps[1], y*steps[0]:(y+1)*steps[0], :] = \
                    self._elements[random.choice(list(normal_element_ids))]
        return image

    def _assign_display_hsb_colors(self, elements):
        """
        RGB colors for element

        Note: Unused
        """
        def get_h_avg(h):
            if h[0] > h[1]:
                return 0
            return h[0] + (h[1] - h[0]) / 2
        self._elements = dict()
        for el in elements:
            self._elements[el["id"]] = \
                hsl2rgb(
                    get_h_avg(el["h"]),
                    el["s"][0] + (el["s"][1] - el["s"][0]) / 2,
                    el["b"][0] + (el["b"][1] - el["b"][0]) / 2
                )


class ColorSetTransformer:
    """

    FIXME: inner_ids vs (element_id, element_name)?

    """

    def __init__(self, elements=COLOR_ELEMENTS):

        self._mapping_inner_id = None
        self._mapping_feature = None
        # self._mapping_display_color = None
        self._element_dtype = np.uint8
        self._compose_low_elements(elements)
        # self._assign_display_colors()

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, single=True):
        if single:
            X_flatten = X.reshape((-1, X.shape[-1]))
            mask = np.zeros(X_flatten.shape[0], dtype=self._element_dtype)
            for i in range(len(X_flatten)):
                mask[i] = self._convert2element(*rgb2hsl(*X_flatten[i]))
            return mask.reshape(X.shape[:-1])
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def filter_elements(self, X, element_ids, single=True):
        """
        Filter elements in provided data.

        Parameters
        ----------
        X : ndarray
            Transformed data that uses inner ids as elements.
        element_ids : list, tuple, set
            Element ids to filter.
        single : bool
            If true (default), then provided X is one object data. Otherwise,
            X contains multiple objects

        Returns
        -------
        mask : ndarray
            Data that contains only elements from element_ids
        """
        mask = np.zeros(X.shape, dtype=self._element_dtype)
        if single:
            for element_id in element_ids:
                indices = X==element_id
                mask[indices] = X[indices]
            return mask
        raise NotImplementedError

    def filter_data(self, X, element_ids, single=True):
        if single:
            X_flatten = X.reshape((-1, X.shape[-1]))
            mask = np.zeros(X_flatten.shape, dtype=X.dtype)
            if isinstance(element_ids, (str, int)):
                for i in range(len(X_flatten)):
                    element_id = self._convert2element(*rgb2hsl(*X_flatten[i]))
                    if element_id == element_ids:
                        mask[i,:] = X_flatten[i,:]
                return mask.reshape(X.shape)
            if isinstance(element_ids, (tuple, list, set)):
                for i in range(len(X_flatten)):
                    element_id = self._convert2element(*rgb2hsl(*X_flatten[i]))
                    if element_id in element_ids:
                        mask[i,:] = X_flatten[i,:]
                return mask.reshape(X.shape)
        raise NotImplementedError

    def transform_to_int(self, X, is_batch=False):
        if is_batch is True:
            pass
        else:
            X_flatten = X.reshape(-1)
            convert = np.vectorize(
                lambda x: self._mapping_inner_id[x] if x in self._mapping_inner_id else 0,
                otypes=(np.uint8,))
            return convert(X_flatten).reshape(X.shape)

    def _compose_low_elements(self, elements):
        self._mapping_feature = dict()
        self._mapping_inner_id = dict()
        if isinstance(elements[0]["id"], str):
            self._element_dtype = "<U10"
        for i, item in enumerate(elements):
            self._mapping_feature[item["id"]] = item
            self._mapping_inner_id[item["id"]] = i+1

    def _convert2element(self, h, s, l):
        for el_id, el in self._mapping_feature.items():
            h_cond = el["h"][0] <= h <= el["h"][1] if el["h"][0] <= el["h"][1] \
                else el["h"][0] <= h <= 240 or 0 <= h <= el["h"][1]
            if h_cond and el["s"][0] <= s <= el["s"][1] and el["b"][0] <= l <= el["b"][1]:
                return el["id"]

    # def _assign_display_colors(self):
    #     """
    #     RGB colors for element
    #
    #     Note: Unused
    #     """
    #     def get_h_avg(h):
    #         if h[0] > h[1]:
    #             return 0
    #         return h[0] + (h[1] - h[0]) / 2
    #     self._mapping_display_color = dict()
    #     for el_id, el in self._mapping_feature.items():
    #         self._mapping_display_color[el_id] = \
    #             hsl2rgb(
    #                 get_h_avg(el["h"]),
    #                 el["s"][0] + (el["s"][1] - el["s"][0]) / 2,
    #                 el["b"][0] + (el["b"][1] - el["b"][0]) / 2
    #             )


def load_hist_elements_from_images(files):
    def extract_color(file):
        try:
            im = Image.open(file).convert()
            pix = im.load()
            return pix[1, 1]
        except:
            warnings.warn("Cannot load this image:", file)
            return None

    colors = dict()
    i = 0
    for file in files:
        color_values = extract_color(file)
        colors[color_values] = "e{}".format(i)
        colors["e{}".format(i)] = color_values
        i += 1
    return colors


def load_hist_elements_from_json(file):
    with open(file) as json_file:
        return json.load(json_file)["elements"]


def convert2hist(image, converter, mode="json"):

    if mode == "json":
        return _convert2hist_from_json(image, converter)
    elif mode == "image":
        return _convert2hist_from_image(image, converter)

    raise Exception("Unknown mode.")


def _convert2hist_from_image(image, pixel_converter):
    pixels = image.load()
    data = []
    for i in range(image.width):
        for j in range(image.height):
            data.append(pixel_converter[pixels[i, j]])
    return Histogram(data, normalized=True, size=image.width*image.height)


def get_rgb_colors(color_elements):
    def get_h_avg(h):
        if h[0] > h[1]:
            return 0
        return h[0] + (h[1] - h[0]) / 2
    converter = dict()
    for el in color_elements:
        converter[el["id"]] = hsl2rgb(get_h_avg(el["h"]),
                                      el["s"][0] + (el["s"][1] - el["s"][0])/2,
                                      el["b"][0] + (el["b"][1] - el["b"][0])/2)
    return converter


def _convert2hist_from_json(image, color_elements, with_other=False):

    def convert2element(h, s, l):
        for el in color_elements:
            h_cond = el["h"][0] <= h <= el["h"][1] if el["h"][0] <= el["h"][1] \
                else el["h"][0] <= h <= 240 or 0 <= h <= el["h"][1]
            if h_cond and el["s"][0] <= s <= el["s"][1] and el["b"][0] <= l <= el["b"][1]:
                return el["id"]

    pixels = image.load()
    data = []
    for i in range(image.width):
        for j in range(image.height):
            el_id = convert2element(*rgb2hsl(*pixels[i, j]))
            if el_id:
                data.append(el_id)
            elif not el_id and with_other:
                el_id = "other"
                data.append(el_id)
            else:
                print("other:", rgb2hsl(*pixels[i, j]))
                pass

    return Histogram(data, normalized=True, size=image.width * image.height)


def convert2hist_1d(image, color_elements, grid_1d):
    def convert2element(h, s, l):
        for el in color_elements:
            h_cond = el["h"][0] <= h <= el["h"][1] if el["h"][0] <= el["h"][1] \
                else el["h"][0] <= h <= 240 or 0 <= h <= el["h"][1]
            if h_cond and el["s"][0] <= s <= el["s"][1] and el["b"][0] <= l <= el["b"][1]:
                return el["id"]

    pixels = image.load()

    position_elements = get_positional_grid_1d(image.width, image.height, grid_1d)

    hist = Histogram1D(data=None)

    for el in position_elements:

        x_start = math.floor(el["pos"][0])
        y_start = math.floor(el["pos"][1])
        x_end = math.floor(el["pos"][2])
        y_end = math.floor(el["pos"][3])

        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                el_id = convert2element(*rgb2hsl(*pixels[i, j]))
                if (el["id"], el_id) not in hist:
                    hist[(el["id"], el_id)] = HElement((el["id"], el_id), 0)
                hist[(el["id"], el_id)].value += 1

    hist.normalize(image.width * image.height)
    return hist


def generate_image(U, color_converter, delta=5, add_normal_color=None, seed=None):

    random.seed(seed)
    color_sample = list(random.choice(U) for _ in range(delta*delta))
    color_elements = [color_converter[el] for el in color_sample]

    img = Image.new("RGB", (100, 100), "black")
    draw = ImageDraw.Draw(img)

    step = int(100 / delta)

    for i in range(delta):
        for j in range(delta):
            draw.rectangle(((i * step, j * step), ((i + 1) * step, (j + 1) * step)),
                           fill="rgb({},{},{})".format(*color_elements[delta*i + j]))

    if not add_normal_color:
        return img

    cx = random.randint(0, delta)
    cy = random.randint(0, delta)

    for _ in range(delta*delta):

        x = round(random.normalvariate(mu=cx, sigma=delta/5))
        y = round(random.normalvariate(mu=cy, sigma=delta/5))

        # if x >= delta: x = delta - 1
        # if y >= delta: y = delta - 1
        #
        # if x < 0: x = 0
        # if y < 0: y = 0

        draw.rectangle(((x * step, y * step), ((x + 1) * step - 1, (y + 1) * step - 1)),
                       fill="rgb({},{},{})".format(*color_converter[random.choice(add_normal_color)]))

    return img


def show_color_elements(elements=None, element_ids="all", element_type="hsb40", title="Low-level elements"):
    if element_type == "hsb40":
        x = np.array(tuple(COLOR_ELEMENTS_RGB.keys()))
        y = np.ones(len(COLOR_ELEMENTS_RGB))
        colors = ["rgb{}".format(rgb) for rgb in COLOR_ELEMENTS_RGB.values()]
        if element_ids != "all":
            # TODO: think to inverse np.any([x == el ...])
            y[np.all([x != el for el in tuple(element_ids)], axis=0)] = 0
        fig = make_subplots(rows=1, cols=1, subplot_titles=None)
        fig.add_bar(x=x, y=y, marker_color=colors, row=1, col=1)
        fig.update_xaxes(gridcolor="#bdbdbd", title="Elements", titlefont=dict(color="grey"), row=1, col=1)
        fig.update_yaxes(visible=False, row=1, col=1)
        fig.update_layout(margin=dict(l=60, r=60, t=60, b=60), plot_bgcolor="#fefefe", showlegend=False,
                          height=160 * 1, width=800, title_text=title)
        fig.show()
        return

    raise NotImplementedError


def plot_position_grid_plotly(position_transformer, fig, row, col, element_ids=None):
    max_x = position_transformer._size[1]
    max_y = position_transformer._size[0]
    for item_id, item in position_transformer.find_elements(element_ids):
        fig.add_shape(
            dict(type="rect",
                 x0=item[1], x1=item[3]+1, y0=max_y-item[0], y1=max_y-item[2]-1,
                 line_color="red",
                 line_width=2,
                 fillcolor="rgba(255,0,0,0)"),
            row=row, col=col,
        )
        fig.add_annotation(
            x=item[1]+(item[3]+1-item[1])*0.5, y=max_y-item[0]-(item[2]+1-item[0])*0.5,
            showarrow=False,
            font=dict(size=16, color="red"),
            text=item_id,
            row=row, col=col)
    fig.update_yaxes(range=[0, max_y], constrain="domain", scaleanchor="x", row=row, col=col)
    fig.update_xaxes(range=[0, max_x], constrain="domain", scaleanchor="y", row=row, col=col)
    return fig


def plot_histogram(elements, values, colors, fig, row, col):
    fig.add_bar(x=elements, y=values, marker_color=colors, width=0.5, row=row, col=col)
    fig.update_xaxes(gridcolor="#bdbdbd", title="Elements", titlefont=dict(color="grey"), row=row, col=col)
    fig.update_yaxes(gridcolor="#bdbdbd", title="Counts", titlefont=dict(color="grey"), row=row, col=col)
    return fig


def show_histogram_(image, HEs, color_converter, position_converter, image_titles=None, hist_titles=None,
                    names=None, main_title="Elements"):
    rows_num = len(HEs)

    subplot_titles = list()

    if image_titles and hist_titles and len(image_titles) == len(hist_titles) == rows_num:
        subplot_titles = sum([[image_titles[i], hist_titles[i]] for i in range(len(image_titles))], [])

    fig = make_subplots(rows=rows_num, cols=2, column_widths=[0.2, 0.8], subplot_titles=subplot_titles)

    for i in range(rows_num):
        img_HE = generate_position_image_with_context(img, get_positional_element_set(HE_list[i]), position_converter)

        HEpc_el, HEpc_val, HEpc_clr = get_data_to_display(HE_list[i], color_converter)

        fig.add_image(z=img_HE, row=i + 1, col=1, name=names[i][0] if names else None)
        fig.add_bar(x=HEpc_el, y=HEpc_val, marker_color=HEpc_clr,
                    width=0.5, row=i + 1, col=2, name=names[i][1] if names else None)
        fig.update_xaxes(gridcolor="#bdbdbd", title="Elements", titlefont=dict(color="grey"), row=i + 1, col=2)
        fig.update_yaxes(gridcolor="#bdbdbd", title="Counts", titlefont=dict(color="grey"), row=i + 1, col=2)

    fig.update_layout(plot_bgcolor="#fefefe", showlegend=False,
                      height=300 * rows_num, width=900, title_text=main_title)
    return fig


def show_histogram(hist_list, U, colors, titles=None, names=None, full=True, main_title="Elements"):
    rows_num = len(hist_list)
    fig = make_subplots(rows=rows_num, cols=1, subplot_titles=titles)

    for i in range(rows_num):
        fig.add_bar(x=U, y=hist_list[i], marker_color=colors, row=i + 1, col=1, name=names[i] if names else names)
        fig.update_xaxes(gridcolor="#bdbdbd", title="Elements", titlefont=dict(color="grey"), row=i + 1, col=1)
        fig.update_yaxes(gridcolor="#bdbdbd", title="Count", titlefont=dict(color="grey"), row=i + 1, col=1)

    fig.update_layout(plot_bgcolor="#fefefe", showlegend=False,
                      height=240 * rows_num, width=800, title_text=main_title)
    return fig


def generate_positional_grid_1d(num_x, num_y):
    elements = list()
    for i in range(num_y):
        for j in range(num_x):
            element = dict()
            element["id"] = "e{}".format(i*num_x + j + 1)
            element["pos"] = (j*1/num_x, i*1/num_y, 1/num_x, 1/num_y)
            elements.append(element)
    return elements


def get_positional_grid_1d(width, height, elements):
    elements_abs = list()
    for el in elements:
        x_start = el["pos"][0] * width
        y_start = el["pos"][1] * height
        x_end = x_start + el["pos"][2] * width
        y_end = y_start + el["pos"][3] * height
        elements_abs.append({"id": el["id"], "pos": (x_start, y_start, x_end, y_end)})
    return elements_abs


def generate_position_image(element, position_converter):
    img = Image.new("RGB", (100, 100), "white")

    if isinstance(element, E):
        Ep = element.value
        elements = Ep.strip(" ()").split("+")
    elif isinstance(element, str):
        if element == "all":
            elements = position_converter.keys()
        else:
            Ep = element
            elements = Ep.strip(" ()").split("+")
    else:
        return img

    draw = ImageDraw.Draw(img)

    for el in elements:
        x_start, y_start, x_end, y_end = position_converter[el]
        draw.rectangle(((x_start, y_start), (x_end, y_end)), fill="gold", outline=True, width=1)
        draw.text((x_start, y_start), el, fill="black")

    draw.rectangle(((0, 0), (100 - 1, 100 - 1)), outline=True, width=1)
    return img


def generate_position_image_with_context(image, elements, position_converter):
    img = image.copy()
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.2)

    # img = Image.new("RGB", (100, 100), "white")

    if len(elements) == 0:
        return img

    draw = ImageDraw.Draw(img)

    for el in elements:
        x_start, y_start, x_end, y_end = position_converter[el]
        box = (int(x_start), int(y_start), x_end - 1, y_end - 1)

        im = image.crop(box)

        img.paste(im, (int(x_start), int(y_start)))

    draw.rectangle(((0, 0), (100 - 1, 100 - 1)), outline=True)
    return img


def get_data_to_display(hist, color_converter):
    hist_elements = sorted(hist.to_dict().items(), key=lambda x: int(x[0][1].strip("e")))

    elements = ["({})".format(",".join(el[0])) for el in hist_elements]
    values = [el[1] for el in hist_elements]
    colors = ["rgb{}".format(color_converter[el[0][1]]) for el in hist_elements]

    return (elements, values, colors)


def get_positional_element_set(element):
    if isinstance(element, HElementSet):
        return {el[0][0] for el in element.to_dict().items()}
    raise Exception("Wrong element type.")


def show_histogram1d(HE_list, img, color_converter, position_converter, image_titles=None, hist_titles=None,
                     names=None, main_title="Elements"):
    rows_num = len(HE_list)

    subplot_titles = list()

    if image_titles and hist_titles and len(image_titles) == len(hist_titles) == rows_num:
        subplot_titles = sum([[image_titles[i], hist_titles[i]] for i in range(len(image_titles))], [])

    fig = make_subplots(rows=rows_num, cols=2, column_widths=[0.2, 0.8], subplot_titles=subplot_titles)

    for i in range(rows_num):
        img_HE = generate_position_image_with_context(img, get_positional_element_set(HE_list[i]), position_converter)

        HEpc_el, HEpc_val, HEpc_clr = get_data_to_display(HE_list[i], color_converter)

        fig.add_image(z=img_HE, row=i + 1, col=1, name=names[i][0] if names else None)
        fig.add_bar(x=HEpc_el, y=HEpc_val, marker_color=HEpc_clr,
                    width=0.5, row=i + 1, col=2, name=names[i][1] if names else None)
        fig.update_xaxes(gridcolor="#bdbdbd", title="Elements", titlefont=dict(color="grey"), row=i + 1, col=2)
        fig.update_yaxes(gridcolor="#bdbdbd", title="Counts", titlefont=dict(color="grey"), row=i + 1, col=2)

    fig.update_layout(plot_bgcolor="#fefefe", showlegend=False,
                      height=300 * rows_num, width=900, title_text=main_title)
    return fig


# Plot complete histogram view
def wrapper_show_complete_histogram():
    def _show_complete_histogram(H, title="Complete Histogram View"):
        complete_values = convert_complete_histogram_values(complete_elements, H)
        fig = make_subplots(rows=1, cols=1, subplot_titles=(title,))
        fig.add_bar(x=complete_elements,
                    y=complete_values,
                    marker_color=complete_colors, row=1, col=1)
        fig.update_xaxes(gridcolor="#bdbdbd", title="Elements", titlefont=dict(color="grey"), row=1, col=1)
        fig.update_yaxes(gridcolor="#bdbdbd", title="Counts", titlefont=dict(color="grey"), row=1, col=1)
        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20),
                          plot_bgcolor="#fefefe", showlegend=False, height=240, width=800)
        fig.show()

    complete_elements = [key for key, _ in sorted(COLOR_ELEMENTS_RGB.items(),
                                                  key=lambda x: int(x[0].lstrip("e")))]

    complete_colors = ["rgb{}".format(COLOR_ELEMENTS_RGB[item]) for item in complete_elements]
    return _show_complete_histogram


show_complete_histogram = wrapper_show_complete_histogram()


# def show_rank_images(images, title):
#     col_num = len(images)
#
#     subplot_titles = list(range(1, col_num + 1))
#
#     fig = make_subplots(rows=1, cols=col_num, subplot_titles=subplot_titles)
#
#     for i in range(col_num):
#         fig.add_image(z=images[i], row=1, col=i + 1)
#     fig.update_layout(plot_bgcolor="#fefefe", width=900, height=300, showlegend=False, title_text=title)
#
#     return fig


def show_rank_images(images, ranked_images, title, limit=11, cols=5):
    img_limit = len(ranked_images) if limit > len(ranked_images) else limit
    row_num = -(-img_limit // cols)

    fig = make_subplots(
        rows=row_num, cols=cols,
        subplot_titles=[
            "rank={}<br>id={}<br>score={:0.4f}".format(
                i + 1,
                image_id,
                image_score) for i, (image_id, image_score) in enumerate(ranked_images)
        ]
    )

    for i in range(row_num):
        for j in range(cols):
            indx = i * cols + j
            if indx >= img_limit:
                break
            else:
                fig.add_image(z=images[ranked_images[indx][0]], row=i + 1, col=j + 1)

    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(
        margin=dict(l=20, r=20, t=100, b=20),
        plot_bgcolor="#fefefe",
        width=900, height=200 * row_num,
        showlegend=False,
        title={
            "text": title,
            "x": 0.5,
            "y": 1.0
        }
    )

    return fig


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