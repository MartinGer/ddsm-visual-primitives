import os
import math
import sys
sys.path.insert(0, '..')
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from PIL import Image, ImageOps
import matplotlib.colors

from db.doctor import insert_doctor_into_db_if_not_exists
from db.database import DB
from training.analyze_single_image import SingleImageAnalysis
from training.common.dataset import get_preview_of_preprocessed_image
from annotations import annotations

STATIC_DIR = 'static'
DATA_DIR = 'data'
LOG_DIR = os.path.join(DATA_DIR, 'log')
ACTIVATIONS_FOLDER = os.path.join(STATIC_DIR, 'activation_maps')
HEATMAPS_FOLDER = os.path.join(STATIC_DIR, 'heatmaps')
PREPROCESSED_IMAGES_FOLDER = os.path.join(STATIC_DIR, 'preprocessed_images')
PREPROCESSED_MASKS_FOLDER = os.path.join(STATIC_DIR, 'preprocessed_masks')

single_image_analysis = None


def init_single_image_analysis(checkpoint_path):
    global single_image_analysis
    single_image_analysis = SingleImageAnalysis(checkpoint_path)
    model = single_image_analysis.get_model()
    return model


def get_survey(name, model, unit):
    db = DB()
    conn = db.get_connection()

    select_unit = unit
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


def store_survey(name, model, unit, shows_phenomena, phenomena):
    db = DB()
    conn = db.get_connection()

    phenomena_description = '\n'.join(phenomena)
    select_unit = int(unit)
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


def register_doctor_if_not_exists(name):
    insert_doctor_into_db_if_not_exists(name)


def normalize_activation_map(activation_map, all_activation_maps=None):
    if all_activation_maps is None:
        all_activation_maps = activation_map
    max_value = all_activation_maps.max()
    min_value = all_activation_maps.min()
    activation_map = (activation_map - min_value) / (max_value - min_value)
    return activation_map


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
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    select_stmt = "SELECT image.image_path, image_unit_activation.activation FROM image_unit_activation " \
                  "INNER JOIN image ON image_unit_activation.image_id = image.id " \
                  "WHERE image_unit_activation.unit_id = ? ORDER BY image_unit_activation.activation DESC " \
                  "LIMIT ?"
    result = c.execute(select_stmt, (unit_id, count))
    top_images = [row[0] for row in result]
    return top_images


def get_top_patches_and_heatmaps_for_unit(unit_id, count):
    unit_id = int(unit_id)
    top_patches = get_top_patches_for_unit(unit_id, count)
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


def get_top_patches_for_unit(unit_id, count, include_normal=False):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    # example patch_filename: images/115/cancer_09-B_3134_1.RIGHT_MLO.LJPEG.1-x1750_y1750_w700_h700_imgfrac0.25_stridefrac0.5.jpg
    # patient id is most of the time characters 12-17
    # FIXME: GROUP BY currently only works for images in folders with same amount of digits (images/xxx)
    if include_normal:
        # get highest activations regardless for which class on patches, including normal ones
        # NOTE: doesn't make a difference at the moment because there are no normal patches in DB
        select_stmt = "SELECT patch_filename, activation, SUBSTR(patch_filename, 11, 16) AS patient FROM patch_unit_activation " \
                      "WHERE unit_id = ? " \
                      "GROUP BY patient " \
                      "ORDER BY MAX(activation) DESC " \
                      "LIMIT ?"
    else:
        # get highest activations regardless for which class on patches that show a tumor
        select_stmt = "SELECT patch_filename, activation, SUBSTR(patch_filename, 11, 16) AS patient  FROM patch_unit_activation " \
                      "WHERE unit_id = ? AND ground_truth != 0 " \
                      "GROUP BY patient " \
                      "ORDER BY MAX(activation) DESC " \
                      "LIMIT ?"

    print("Query database for top patches of unit {}...".format(unit_id))
    result = c.execute(select_stmt, (unit_id, count))
    top_patches = [row[0] for row in result]
    return top_patches


def get_appearances_in_top_units(unit_id, class_id):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    select_stmt = "SELECT appearances_in_top_units FROM unit_class_influence " \
                  "WHERE unit_id = ? AND class_id = ?"
    print("Query database for appearances_in_top_units of unit {}...".format(unit_id))
    result = c.execute(select_stmt, (unit_id, class_id))
    result = [row[0] for row in result]
    if not result:
        return 0
    return result[0]


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

    return heatmap_paths, preprocessed_size


def generate_phenomenon_heatmap(forward_pass_result, annotation_id, preprocessed_size, user, net_id):
    units = get_units_with_annotation(annotation_id, user, net_id)
    activation_maps = np.asarray([forward_pass_result.feature_maps[unit_id - 1] for unit_id in units])
    combined_activations = np.amax(activation_maps, axis=0)

    heatmap = _activation_map_to_heatmap(combined_activations, activation_maps)
    heatmap = heatmap.resize(preprocessed_size, resample=Image.BICUBIC)

    heatmap_path = os.path.join(HEATMAPS_FOLDER,
                                '{}_{}_{}.jpg'.format(forward_pass_result.image_path.split('/')[-1][:-4],
                                                      annotation_id, net_id))
    heatmap.save(heatmap_path, "JPEG")
    return heatmap_path


def get_units_with_annotation(annotation_id, user, net_id):
    db = DB()
    conn = db.get_connection()

    select_stmt = "SELECT unit_id FROM unit_annotation " \
                  "WHERE descriptions LIKE ? " \
                  "AND net_id = (SELECT id FROM net WHERE net = ?) " \
                  "AND doctor_id = (SELECT id FROM doctor WHERE name = ?);"

    like_string = '%{}%'.format(annotation_id)
    result = conn.execute(select_stmt, (like_string, net_id, user))
    units = [row[0] for row in result]
    return units


def survey2unit_annotations_ui(survey, language):
    if survey:
        shows_phenomena, descriptions = survey
        if not shows_phenomena:
            descriptions = ["✗ No phenomena"]
    else:
        descriptions = ["Not annotated"]

    localized_descriptions = []
    for description in descriptions:
        localized_ann = annotations[language].get(description)
        if localized_ann:
            localized_descriptions.append(localized_ann)
        else:
            localized_descriptions.append(description)

    return localized_descriptions


def human_readable_annotation(annotation_id, language):
    return annotations[language].get(annotation_id, annotation_id)


def get_correct_classified_images(class_id, count):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    select_stmt = "SELECT image.image_path FROM image " \
                  "INNER JOIN image_classification ON image_classification.image_id = image.id " \
                  "WHERE image.ground_truth = image_classification.class_id AND image.ground_truth = ? " \
                  "ORDER BY image.image_path ASC " \
                  "LIMIT ?"
    result = c.execute(select_stmt, (class_id, count))
    images = [row[0].split("/")[-1] for row in result]
    return images


def similarity_metric(image_filename, analysis_result, name, model):
    reference_image = _get_image_id(image_filename)
    top_units_and_activations = analysis_result.get_top_units(analysis_result.classification, 10)
    annotated_units = _get_annotated_units(name, model)
    annotated_top_units = [item[0] + 1 for item in top_units_and_activations if item[0] + 1 in annotated_units]
    ranks_of_units_per_image = {}

    for unit_id in annotated_top_units:
        for image_id, rank in _get_ranks_of_unit(unit_id, analysis_result.classification, model):
            if image_id in ranks_of_units_per_image:
                ranks_of_units_per_image[image_id].append(rank)
            else:
                ranks_of_units_per_image[image_id] = [rank]

    reference_ranks = ranks_of_units_per_image[reference_image]
    del ranks_of_units_per_image[reference_image]

    similarities = []

    for image_id in ranks_of_units_per_image.keys():
        ranks = ranks_of_units_per_image[image_id]
        similarity = cosine_distance(reference_ranks, ranks)
        similarities.append((image_id, similarity))

    similarities.sort(key=lambda x: x[1], reverse=False)

    top20_images = [image_id for image_id, similarity in similarities[:20]]
    top20_image_paths = [_get_image_path(image_id) for image_id in top20_images]

    ground_truth_of_top20 = np.asarray([_get_ground_truth(image_id) for image_id in top20_images])

    return np.unique(ground_truth_of_top20, return_counts=True)[1], top20_image_paths


def _get_image_id(image_path):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    select_stmt = "SELECT id FROM image " \
                  "WHERE image_path = ?;"
    c.execute(select_stmt, (image_path,))
    result = c.fetchone()[0]
    return result


def _get_image_path(image_id):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    select_stmt = "SELECT image_path FROM image " \
                  "WHERE id = ?;"
    c.execute(select_stmt, (image_id,))
    result = c.fetchone()[0]
    return result


def _get_annotated_units(name, model):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()

    select_net = "(SELECT id FROM net WHERE net = '{}')".format(model)
    select_doctor = "(SELECT id FROM doctor WHERE name = '{}')".format(name)

    select_stmt = "SELECT unit_id FROM unit_annotation " \
                  "WHERE net_id = {net} " \
                  "AND doctor_id = {doctor} " \
                  "AND unit_annotation.shows_concept = 1;".format(doctor=select_doctor, net=select_net)
    result = c.execute(select_stmt)
    annotated_units = [row[0] for row in result]

    return annotated_units


def _get_activations_for_unit(unit_id, classification, model):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()

    select_net = "(SELECT id FROM net WHERE net = '{}')".format(model)

    select_stmt = "SELECT image_id, activation FROM image_unit_activation " \
                  "WHERE net_id = {net} " \
                  "AND class_id = ? " \
                  "AND unit_id = ?;".format(net=select_net)
    result = c.execute(select_stmt, (classification, unit_id))
    activations = [(row[0], row[1]) for row in result]

    return activations


def _get_ranks_of_unit(unit_id, classification, model):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()

    select_net = "(SELECT id FROM net WHERE net = '{}')".format(model)

    select_stmt = "SELECT image_id, rank FROM image_unit_activation " \
                  "WHERE net_id = {net} " \
                  "AND class_id = ? " \
                  "AND unit_id = ?;".format(net=select_net)
    result = c.execute(select_stmt, (classification, unit_id))
    activations = [(row[0], row[1]) for row in result]

    return activations


def _get_ground_truth(image_id):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    select_stmt = "SELECT ground_truth FROM image " \
                  "WHERE id = ?;"
    c.execute(select_stmt, (image_id,))
    result = c.fetchone()[0]
    return result
