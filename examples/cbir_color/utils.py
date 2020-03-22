import colorsys
import json
import sys
import warnings
import random

from PIL import Image, ImageDraw
from plotly.subplots import make_subplots

sys.path.insert(0, "../../")
from lshist.histogram import Histogram


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


def rgb2hsl(r, g, b):
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return h * 240, s * 240, l * 240


def hsl2rgb(h, s, b):
    r, g, b = colorsys.hls_to_rgb(h/240, b/240, s/240)
    return round(255 * r), round(255 * g), round(255 * b)


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


def generate_image(U, color_converter, seed=None):

    random.seed(seed)
    color_sample = list(random.choice(U) for _ in range(25))
    color_elements = [color_converter[el] for el in color_sample]

    img = Image.new("RGB", (100, 100), "black")
    draw = ImageDraw.Draw(img)
    for i in range(5):
        for j in range(5):
            draw.rectangle(((i * 20, j * 20), ((i + 1) * 20, (j + 1) * 20)),
                           fill="rgb({},{},{})".format(*color_elements[5*i + j]))
    return img


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
