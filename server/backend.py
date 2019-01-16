import base64
import glob
import os
import pickle
import uuid
import math
import sys
sys.path.insert(0, '..')
from db.doctor import insert_doctor_into_db_if_not_exists
from db.database import DB
import numpy as np
from PIL import Image
import matplotlib.colors

from training.grad_cam import run_grad_cam
from training.analyze_single_image import SingleImageAnalysis
from training.common.dataset import get_preview_of_preprocessed_image

STATIC_DIR = 'static'
DATA_DIR = 'data'
LOG_DIR = os.path.join(DATA_DIR, 'log')
ACTIVATIONS_FOLDER = os.path.join(STATIC_DIR, 'activation_maps')

DB_FILENAME = os.environ['DB_FILENAME'] if 'DB_FILENAME' in os.environ else 'test.db'

single_image_analysis = SingleImageAnalysis()


def get_models_and_layers(full=False, ranked=False):
    if full:
        unit_vis_dir = os.path.join(STATIC_DIR, 'unit_vis')
    else:
        unit_vis_dir = os.path.join(STATIC_DIR, 'unit_vis_subset')
    models_and_layers = []
    models = sorted(os.listdir(unit_vis_dir))
    for model in models:
        layers = sorted(os.listdir(os.path.join(unit_vis_dir, model)))
        for layer in layers:
            rankings_path = 'data/unit_rankings/{}/{}/rankings.pkl'.format(model, layer)
            if not ranked or os.path.exists(rankings_path):
                models_and_layers.append((model, layer))
    return models_and_layers


def get_responded_units(name):
    responses = get_responses(name)
    all_units = []
    for (model, layer) in get_models_and_layers(full=True):
        layer_dir = os.path.join('unit_vis', model, layer)
        model_and_layer = '{}/{}'.format(model, layer)
        units = []
        for unit in sorted(os.listdir(os.path.join(STATIC_DIR, layer_dir))):
            key = '{}/{}/{}'.format(model, layer, unit)
            if key in responses.keys():
                units.append(unit)
        if len(units) > 0:
            all_units.append((model_and_layer, units))
    return all_units


def get_units(name, model, layer, sample=8, full=False, ranked=False):
    if full:
        layer_dir = os.path.join('unit_vis', model, layer)
    else:
        layer_dir = os.path.join('unit_vis_subset', model, layer)
    responses = get_responses(name)
    units = []
    sums = []
    labels = get_labels()
    label_symbols = {0: '-', 1: 'o', 2: '+'}
    for unit in sorted(os.listdir(os.path.join(STATIC_DIR, layer_dir))):
        key = '{}/{}/{}'.format(model, layer, unit)
        message = '[response recorded]' if key in responses else ''
        unit_dir = os.path.join(layer_dir, unit)
        image_names = sorted(os.listdir(os.path.join(STATIC_DIR, unit_dir)))[:sample]
        unit_labels = [labels[image_name[5:]] for image_name in image_names]
        unit_labels_str = ' '.join([label_symbols[label] for label in unit_labels])
        image_paths = [os.path.join(unit_dir, x) for x in image_names]
        units.append((unit, message, image_paths, unit_labels_str))
        sums.append(sample - sum(unit_labels))
    if ranked:
        num_units_per_class = 20
        ranked_units = []
        rankings_path = 'data/unit_rankings/{}/{}/rankings.pkl'.format(model, layer)
        with open(rankings_path, 'rb') as f:
            rankings = pickle.load(f)
        for class_index, unit_rankings in enumerate(rankings):
            for unit_index, count in unit_rankings[:num_units_per_class]:
                unit = units[unit_index]
                ranked_units.append((unit[0], unit[1], unit[2],
                                     '(class {}, count {}) {}'.format(class_index, count, unit[3])))
        units = ranked_units
    else:
        sums, units = zip(*sorted(zip(sums, units)))
    return list(units)


def get_unit_data(name, model, layer, unit, sample=32, num_cols=4):
    unit_dir = os.path.join('unit_vis', model, layer, unit)
    image_names = sorted(os.listdir(os.path.join(STATIC_DIR, unit_dir)))[:sample]
    entries = []
    labels = get_labels()
    label_names = {0: 'normal', 1: 'begnin', 2: 'malignant'}
    for image_name in image_names:
        # remove the first 5 chars containing the image rank
        parts = image_name[5:].split('-')
        label = label_names[labels[image_name[5:]]]
        raw_name = '{}-{}.jpg'.format(parts[0], parts[1])
        raw_width, raw_height = Image.open(os.path.join(STATIC_DIR, 'raw/{}'.format(raw_name))).size
        # noinspection PyTypeChecker
        box = parts[2].split('_')
        x = int(box[1][1:])
        y = int(box[0][1:])
        w = int(box[3][1:])
        h = int(box[2][1:])
        assert w == h
        height = 100 * float(h) / raw_height
        width = 100 * float(w) / raw_width
        left = 100 * float(x) / raw_width
        top = 100 * float(y) / raw_height
        style = 'position: absolute; height: {}%; width: {}%; left: {}%; top: {}%;'.format(height, width, left, top)
        entry = {
            'img_name': 'unit_vis/{}/{}/{}/{}'.format(model, layer, unit, image_name),
            'style': style,
            'raw_name': 'raw/{}'.format(raw_name),
            'label': label
        }
        entries.append(entry)
    data = [entries[i:i + num_cols] for i in range(0, len(entries), num_cols)]
    responses = get_responses(name)
    old_response = None
    key = '{}/{}/{}'.format(model, layer, unit)
    if key in responses:
        old_response = responses[key]
    return data, old_response


def get_labels():
    labels = {}
    for labels_path in glob.glob('data/labels/*.pickle'):
        print(labels_path)
        with open(labels_path, 'rb') as f:
            labels.update(pickle.load(f))
    return labels


def get_responses(name):
    encoded_name = base64.urlsafe_b64encode(bytes(name, 'utf-8'))
    data_path = os.path.join(DATA_DIR, '{}.pickle'.format(encoded_name))
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            responses = pickle.load(f)
    else:
        responses = {}
    return responses


def get_num_responses(name):
    return len(get_responses(name).keys())


def get_survey(name, model, layer, unit):
    db = DB(DB_FILENAME, '../db/')
    conn = db.get_connection()

    select_unit = int(unit.split("_")[1])  # looks like: unit_0076
    select_net = "(SELECT id FROM net WHERE net = '{}')".format(model)
    select_doctor = "(SELECT id FROM doctor WHERE name = '{}')".format(name)

    select_stmt = "SELECT descriptions FROM unit_annotation " \
                  "WHERE unit_id = {} " \
                  "AND net_id = {} " \
                  "AND doctor_id = {};".format(select_unit, select_net, select_doctor)

    result = conn.execute(select_stmt)
    description = [row for row in result]
    if description:
        description = description[0][0].split('\n')
    return description


def store_survey(name, model, layer, unit, shows_phenomena, phenomena):
    db = DB(DB_FILENAME, '../db/')
    conn = db.get_connection()

    phenomena_description = '\n'.join(phenomena)
    select_unit = int(unit.split("_")[1])  # looks like: unit_0076
    select_net = "(SELECT id FROM net WHERE net = '{}')".format(model)
    select_doctor = "(SELECT id FROM doctor WHERE name = '{}')".format(name)

    if shows_phenomena == "true":
        select_concept = 1
    else:
        select_concept = 0
        phenomena_description = ""

    # try updating in case it exists already
    update_stmt = "UPDATE unit_annotation SET descriptions = '{}', shows_concept={} WHERE " \
                  "unit_id = {} AND net_id = {} AND doctor_id = {};".format(phenomena_description,
                                                                            select_concept,
                                                                            select_unit,
                                                                            select_net,
                                                                            select_doctor)
    conn.execute(update_stmt)
    # make sure it exists
    insert_stmt = "INSERT OR IGNORE INTO unit_annotation(unit_id, net_id, doctor_id, descriptions, shows_concept) " \
                  "VALUES ({}, {}, {}, '{}', {});".format(select_unit,
                                                        select_net,
                                                        select_doctor,
                                                        phenomena_description,
                                                        select_concept)
    conn.execute(insert_stmt)
    conn.commit()


def get_summary():
    summary = []
    for pickle_path in sorted(glob.glob(os.path.join(DATA_DIR, '*.pickle'))):
        with open(pickle_path, 'rb') as f:
            responses = pickle.load(f)
        name = responses.itervalues().next()['name']
        responded_units = get_responded_units(name)
        summary.append((name, responded_units))
    return summary


def register_doctor_if_not_exists(name):
    insert_doctor_into_db_if_not_exists(name, DB_FILENAME, '../db/')


    # resize activation map to img size
def resize_activation_map(img, activation_map):
    basewidth = img.size[0]
    wpercent = (basewidth / float(len(activation_map[0])))
    hsize = int((float(len(activation_map[1])) * float(wpercent)))
    return np.resize(activation_map, (basewidth, hsize))


def normalize_activation_map(activation_map):
    max_value = activation_map.max()
    min_value = activation_map.min()
    activation_map = (activation_map - min_value) / (max_value - min_value)
    return activation_map


def grad_cam():
    run_grad_cam(image_path='static/processed/benign.jpg', cuda=False)


def get_top_images_for_unit(unit_id, count):
    db = DB(DB_FILENAME, '../db/')
    conn = db.get_connection()
    c = conn.cursor()

    select_stmt = "SELECT image.image_path FROM image_unit_activation " \
                  "INNER JOIN image ON image_unit_activation.image_id = image.id " \
                  "WHERE image_unit_activation.unit_id = ? ORDER BY image_unit_activation.activation DESC " \
                  "LIMIT ?"

    top_images = []

    for row in c.execute(select_stmt, (unit_id, count)):
        top_images.append(row[0])

    return top_images


def get_top_images_with_activation_for_unit(unit_id, count):
    if not os.path.exists(ACTIVATIONS_FOLDER):
        os.makedirs(ACTIVATIONS_FOLDER)
    unit_id = int(unit_id)
    top_images = get_top_images_for_unit(unit_id, count)
    preprocessed_top_images = []
    activation_maps = []

    for i, image_path in enumerate(top_images):
        preprocessed_image = get_preview_of_preprocessed_image(os.path.join("../data/ddsm_raw/", image_path))
        preprocessed_image_path = os.path.join(ACTIVATIONS_FOLDER, 'preprocessed_{}.jpg'.format(uuid.uuid4()))
        preprocessed_image.save(preprocessed_image_path)
        preprocessed_top_images.append(preprocessed_image_path)
        act_map_img = get_activation_map(os.path.join("../data/ddsm_raw/", image_path), unit_id)
        activation_map_path = os.path.join(ACTIVATIONS_FOLDER, 'activation_{}.jpg'.format(uuid.uuid4()))
        act_map_img.save(activation_map_path, "JPEG")
        activation_maps.append(activation_map_path)

    return (top_images, preprocessed_top_images, activation_maps)


def get_activation_map(image_path, unit_id):
    preprocessed_full_image = get_preview_of_preprocessed_image(image_path)
    result = single_image_analysis.analyze_one_image(image_path)

    activation_map = result.feature_maps[unit_id]
    act_map_img = to_heatmap(activation_map)
    act_map_img = act_map_img.resize(preprocessed_full_image.size, resample=Image.BICUBIC)
    return act_map_img


def to_heatmap(activation_map):
    activation_map_normalized = normalize_activation_map(activation_map)

    get_highest_activations_in_percentage(activation_map_normalized, 0.25)

    activation_heatmap = np.ndarray((activation_map.shape[0], activation_map.shape[1], 3), np.double)
    for x in range(activation_map.shape[0]):
        for y in range(activation_map.shape[1]):
            v = activation_map_normalized[x, y]
            activation_heatmap[x, y] = matplotlib.colors.hsv_to_rgb((0.5 - (v * 0.5), 1, v))

    activation_heatmap = activation_heatmap * 255
    return Image.fromarray(activation_heatmap.astype(np.uint8), mode="RGB")


def get_highest_activations_in_percentage(activation_map, percentage):
    no_of_elements_in_matrix = activation_map.shape[0] * activation_map.shape[1]
    no_of_elements_in_percentage_range = math.ceil((no_of_elements_in_matrix/100) * percentage)

    flat = activation_map.flatten()
    flat.sort()
    threshold = flat[-no_of_elements_in_percentage_range:][0]

    for x in range(len(activation_map)):
        for y in range(len(activation_map[0])):
            if activation_map[x][y] < threshold:
                activation_map[x][y] = 0

    print('Showing top', percentage, 'percent of activations in activation map. That`s'
          , no_of_elements_in_percentage_range, 'of', no_of_elements_in_matrix, 'elements.')
    return activation_map


# after resize WIP
def get_highest_activations_in_percentage_after_resize(activation_map, percentage):
    no_of_elements_in_matrix = activation_map.size[0] * activation_map.size[1]
    no_of_elements_in_percentage_range = math.ceil((no_of_elements_in_matrix/100) * percentage)
    print(no_of_elements_in_matrix)
    flat = np.array(activation_map)
    flat.sort()
    threshold = flat[-no_of_elements_in_percentage_range:][0]

    print("threshold")
    print(threshold)
    print("activation map")
    print(activation_map[50][50])

    for x in range(activation_map.size[0]):
        for y in range(activation_map.size[1]):
            if activation_map[x, y] < threshold:
                activation_map[x, y] = 0

    print('Showing top', percentage, 'percent of activations in activation map. That`s'
          , no_of_elements_in_percentage_range, 'of', no_of_elements_in_matrix, 'elements.')
    return activation_map


def sum_matrix(matrix):
        return sum(map(sum, matrix))

