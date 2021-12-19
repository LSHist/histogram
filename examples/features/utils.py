import colorsys
import json
import sys
import warnings
import random
import math

from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from plotly.subplots import make_subplots

sys.path.insert(0, "../../")
from histogram.histogram import Histogram, HElementSet, Histogram1D, HElement
from histogram.utils import E


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


def show_rank_images(images, title):
    col_num = len(images)

    subplot_titles = list(range(1, col_num + 1))

    fig = make_subplots(rows=1, cols=col_num, subplot_titles=subplot_titles)

    for i in range(col_num):
        fig.add_image(z=images[i], row=1, col=i + 1)
    fig.update_layout(plot_bgcolor="#fefefe", width=900, height=300, showlegend=False, title_text=title)

    return fig