import base64
import glob
import os
import pickle
import math
import sys
sys.path.insert(0, '..')
import numpy as np
from PIL import Image, ImageOps
import matplotlib.colors

from db.doctor import insert_doctor_into_db_if_not_exists
from db.database import DB
from training.grad_cam import run_grad_cam
from training.analyze_single_image import SingleImageAnalysis
from training.common.dataset import get_preview_of_preprocessed_image

STATIC_DIR = 'static'
DATA_DIR = 'data'
LOG_DIR = os.path.join(DATA_DIR, 'log')
ACTIVATIONS_FOLDER = os.path.join(STATIC_DIR, 'activation_maps')
HEATMAPS_FOLDER = os.path.join(STATIC_DIR, 'heatmaps')
PREPROCESSED_IMAGES_FOLDER = os.path.join(STATIC_DIR, 'preprocessed_images')
PREPROCESSED_MASKS_FOLDER = os.path.join(STATIC_DIR, 'preprocessed_masks')

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

    select_stmt = "SELECT shows_concept, descriptions FROM unit_annotation " \
                  "WHERE unit_id = {} " \
                  "AND net_id = {} " \
                  "AND doctor_id = {};".format(select_unit, select_net, select_doctor)

    result = conn.execute(select_stmt)
    row = [r for r in result]
    if row:
        shows_phenomena, description = row[0]
        if description:
            description = description.split('\n')
        shows_phenomena = True if shows_phenomena == 1 else False
        return shows_phenomena, description
    else:
        return None


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


def resize_activation_map(img, activation_map):
    # resize activation map to img size
    basewidth = img.size[0]
    wpercent = (basewidth / float(len(activation_map[0])))
    hsize = int((float(len(activation_map[1])) * float(wpercent)))
    return np.resize(activation_map, (basewidth, hsize))


def normalize_activation_map(activation_map, all_activation_maps=None):
    if all_activation_maps is None:
        all_activation_maps = activation_map
    max_value = all_activation_maps.max()
    min_value = all_activation_maps.min()
    activation_map = (activation_map - min_value) / (max_value - min_value)
    return activation_map


def grad_cam():
    run_grad_cam(image_path='static/processed/benign.jpg', cuda=False)


def get_top_images_and_heatmaps_for_unit(unit_id, count):
    unit_id = int(unit_id)
    top_images = _get_top_images_for_unit(unit_id, count)
    if len(top_images) < 1:
        print("No top images for unit found, is the database populated?")
        return [], [], []

    if not os.path.exists(HEATMAPS_FOLDER):
        os.makedirs(HEATMAPS_FOLDER)
    if not os.path.exists(PREPROCESSED_IMAGES_FOLDER):
        os.makedirs(PREPROCESSED_IMAGES_FOLDER)

    preprocessed_image_paths = [get_preprocessed_image_path(full_image_name) for full_image_name in top_images]

    checkpoint_identifier = single_image_analysis.checkpoint_path[:-8].replace("/", "_").replace(".", "_")
    heatmap_paths = [os.path.join(HEATMAPS_FOLDER, '{}_{}_{}.jpg'.format(full_image_name[:-4], unit_id, checkpoint_identifier)) for full_image_name in top_images]

    for heatmap_path in heatmap_paths:
        if not os.path.exists(heatmap_path):
            # -> at least one heatmap is missing, regenerate all:
            model_results = [single_image_analysis.analyze_one_image(os.path.join("../data/ddsm_raw/", full_image_name)) for full_image_name in top_images]
            activation_maps = np.asarray([result.feature_maps[unit_id - 1] for result in model_results])
            preprocessed_size = get_preview_of_preprocessed_image(os.path.join("../data/ddsm_raw/", top_images[0])).size

            for i, full_image_name in enumerate(top_images):
                heatmap = _activation_map_to_heatmap(activation_maps[i], activation_maps)
                heatmap = heatmap.resize(preprocessed_size, resample=Image.BICUBIC)
                heatmap.save(heatmap_paths[i], "JPEG")
            break

    return top_images, preprocessed_image_paths, heatmap_paths


def _get_top_images_for_unit(unit_id, count):
    db = DB(DB_FILENAME, '../db/')
    conn = db.get_connection()
    c = conn.cursor()
    select_stmt = "SELECT image.image_path FROM image_unit_activation " \
                  "INNER JOIN image ON image_unit_activation.image_id = image.id " \
                  "WHERE image_unit_activation.unit_id = ? ORDER BY image_unit_activation.activation DESC " \
                  "LIMIT ?"
    result = c.execute(select_stmt, (unit_id, count))
    top_images = [row[0] for row in result]
    return top_images


def get_top_patches_and_heatmaps_for_unit(unit_id, count):
    unit_id = int(unit_id)
    top_patches = _get_top_patches_for_unit(unit_id, count)
    if len(top_patches) < 1:
        print("No top patches for unit found, is the database populated?")
        return [], [], []

    if not os.path.exists(HEATMAPS_FOLDER):
        os.makedirs(HEATMAPS_FOLDER)

    checkpoint_identifier = single_image_analysis.checkpoint_path[:-8].replace("/", "_").replace(".", "_")
    heatmap_paths = [os.path.join(HEATMAPS_FOLDER, '{}_{}_{}.jpg'.format(patch_filename[:-4].replace("/", "_"), unit_id, checkpoint_identifier)) for patch_filename in top_patches]

    for heatmap_path in heatmap_paths:
        if not os.path.exists(heatmap_path):
            # -> at least one heatmap is missing, regenerate all:
            model_results = [single_image_analysis.analyze_one_patch(os.path.join("../data/ddsm_3class/", patch_filename)) for patch_filename in top_patches]
            activation_maps = np.asarray([result.feature_maps[unit_id - 1] for result in model_results])
            patch_size = Image.open(os.path.join("../data/ddsm_3class/", top_patches[0])).size

            for i, full_image_name in enumerate(top_patches):
                heatmap = _activation_map_to_heatmap(activation_maps[i], activation_maps)
                heatmap = heatmap.resize(patch_size, resample=Image.BICUBIC)
                heatmap.save(heatmap_paths[i], "JPEG")
            break

    return top_patches, heatmap_paths


def _get_top_patches_for_unit(unit_id, count):
    db = DB(DB_FILENAME, '../db/')
    conn = db.get_connection()
    c = conn.cursor()
    # get highest activations regardless for which class on patches that aren't normal
    select_stmt = "SELECT DISTINCT patch_filename FROM patch_unit_activation " \
                  "WHERE unit_id = ? AND ground_truth != 0 ORDER BY activation DESC " \
                  "LIMIT ?"
    print("Query database for top patches...")
    result = c.execute(select_stmt, (unit_id, count))
    top_patches = [row[0] for row in result]
    print("Query finished.")
    return top_patches


def get_preprocessed_image_path(full_image_name, root="../data/ddsm_raw/"):
    path = os.path.join(PREPROCESSED_IMAGES_FOLDER, '{}.jpg'.format(full_image_name[:-4]))
    if not os.path.exists(path):
        preprocessed_image = get_preview_of_preprocessed_image(os.path.join(root, full_image_name))
        preprocessed_image.save(path)
    return path


def _activation_map_to_heatmap(activation_map, all_activation_maps):
    activation_map_normalized = normalize_activation_map(activation_map, all_activation_maps)

    #_get_highest_activations_in_percentage(activation_map_normalized, 0.25)

    activation_heatmap = np.ndarray((activation_map.shape[0], activation_map.shape[1], 3), np.double)
    for x in range(activation_map.shape[0]):
        for y in range(activation_map.shape[1]):
            v = activation_map_normalized[x, y]
            activation_heatmap[x, y] = matplotlib.colors.hsv_to_rgb((0.5 - (v * 0.5), 1, v))

    activation_heatmap = activation_heatmap * 255
    return Image.fromarray(activation_heatmap.astype(np.uint8), mode="RGB")


def _get_highest_activations_in_percentage(activation_map, percentage):
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
def _get_highest_activations_in_percentage_after_resize(activation_map, percentage):
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


def _sum_matrix(matrix):
        return sum(map(sum, matrix))


def get_preprocessed_mask_path(image_filename):
    mask_dirs = ["benigns", "benign_without_callbacks", "cancers"]
    for mask_dir in mask_dirs:
        mask_path = os.path.join('../data/ddsm_masks/3class', mask_dir, image_filename[:-4] + '.png')
        if os.path.exists(mask_path):
            if not os.path.exists(PREPROCESSED_MASKS_FOLDER):
                os.makedirs(PREPROCESSED_MASKS_FOLDER)
            preprocessed_mask_path = os.path.join(PREPROCESSED_MASKS_FOLDER, '{}.jpg'.format(image_filename[:-4]))
            if not os.path.exists(preprocessed_mask_path):
                mask_preprocessed = get_preview_of_preprocessed_image(mask_path)
                mask_preprocessed = ImageOps.colorize(ImageOps.equalize(mask_preprocessed), (0, 0, 0), (255, 0, 0))
                mask_preprocessed.save(preprocessed_mask_path)
            return preprocessed_mask_path
    return ""


def get_heatmap_paths_for_top_units(image_filename, top_units_and_activations, units_to_show, root="../data/ddsm_raw/"):
    activation_maps = np.asarray([top_units_and_activations[i][2] for i in range(units_to_show)])

    checkpoint_identifier = single_image_analysis.checkpoint_path[:-8].replace("/", "_").replace(".", "_")
    preprocessed_size = get_preview_of_preprocessed_image(os.path.join(root, image_filename)).size
    heatmap_paths = []

    for i in range(units_to_show):
        heatmap = _activation_map_to_heatmap(activation_maps[i], activation_maps)
        heatmap = heatmap.resize(preprocessed_size, resample=Image.BICUBIC)

        heatmap_path = os.path.join(HEATMAPS_FOLDER,
                                    '{}_{}_{}.jpg'.format(image_filename[:-4], top_units_and_activations[i][0],
                                                          checkpoint_identifier))
        heatmap.save(heatmap_path, "JPEG")
        heatmap_paths.append(heatmap_path)

    return heatmap_paths
