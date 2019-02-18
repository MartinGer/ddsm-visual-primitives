import sys
from functools import lru_cache

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from tqdm import tqdm

sys.path.insert(0, '..')
from db.database import DB


METRIC_CALCULATION_RUNNING = False


def similarity_metric(image_filename, name, model):
    reference_image_id = _get_image_id(image_filename)
    gt_distribution, top20_image_ids, ground_truth_of_top20 = _get_similarity_score_by_image_id(reference_image_id, name, model)
    top20_image_paths = [_get_image_path(image_id) for image_id in top20_image_ids]
    return gt_distribution, top20_image_paths, ground_truth_of_top20


def print_all_similarity_scores(name, model):
    global METRIC_CALCULATION_RUNNING
    if METRIC_CALCULATION_RUNNING:
        return
    METRIC_CALCULATION_RUNNING = True

    print("overall_network_performance_on_annotated_units started")

    for units_to_compare in ('top_10_units', 'top_annotated', 'all_annotated'):
        for feature_to_compare in ('activation', 'rank'):
            _print_average_similarity_score(name, model, units_to_compare, feature_to_compare, image_count=20)

    METRIC_CALCULATION_RUNNING = False


def _print_average_similarity_score(name, model, units_to_compare, feature_to_compare, image_count):
    print("_get_average_similarity_score started:", units_to_compare, feature_to_compare, image_count)
    image_ids = _get_all_image_ids()
    similar_imgs_with_same_gt = 0

    for image_id in tqdm(image_ids):
        gt = _get_ground_truth(image_id)
        gt_distribution = _get_similarity_score_by_image_id(image_id, name, model, units_to_compare, feature_to_compare, image_count)[0]
        similar_imgs_with_same_gt += gt_distribution[gt]

    avg_imgs_with_same_gt = similar_imgs_with_same_gt / len(image_ids)
    print("Avg. similar images with same ground truth: {} of {}".format(avg_imgs_with_same_gt, 20))


def _get_similarity_score_by_image_id(reference_image_id, name, model, units_to_compare='top_10_units', feature_to_compare='activation', image_count=20):
    classification = _get_classification(reference_image_id, model)

    if units_to_compare == 'top_10_units':
        _units_to_compare = _get_top_n_units(reference_image_id, classification, model, 10)
    elif units_to_compare == 'top_annotated':
        _units_to_compare = _get_annotated_top_units(reference_image_id, classification, name, model)
    elif units_to_compare == 'all_annotated':
        _units_to_compare = _get_all_annotated_units(name, model)
    else:
        raise ValueError("Unknown content of units_to_compare.")

    if not _units_to_compare:
        print("No units to compare for this image -> can't find similar images.")
        return (0, 0, 0), [], []

    if feature_to_compare == 'activation':
        top_image_ids = _get_similar_images_by_activations(reference_image_id, classification, model, _units_to_compare, image_count)
    elif feature_to_compare == 'rank':
        top_image_ids = _get_similar_images_by_ranks(reference_image_id, classification, model, _units_to_compare, image_count)
    else:
        raise ValueError("Unknown content of feature_to_compare.")

    ground_truth_of_top_images = np.asarray([_get_ground_truth(image_id) for image_id in top_image_ids])
    gt_distribution = ((ground_truth_of_top_images == 0).sum(), (ground_truth_of_top_images == 1).sum(), (ground_truth_of_top_images == 2).sum())

    return gt_distribution, top_image_ids, ground_truth_of_top_images


def _get_annotated_top_units(reference_image_id, classification, name, model):
    annotated_units = _get_all_annotated_units(name, model)
    ranks_of_units_per_image = _get_unit_ranks(annotated_units, classification, model)
    annotated_top_units = [annotated_units[i] for i in range(len(annotated_units)) if
                           ranks_of_units_per_image[reference_image_id][i] < 10]
    return annotated_top_units


def _get_similar_images_by_ranks(reference_image_id, classification, model, units_to_compare, count):
    ranks_of_units_per_image = _get_unit_ranks(units_to_compare, classification, model)
    return _get_similar_images(reference_image_id, ranks_of_units_per_image, count)


def _get_similar_images_by_activations(reference_image_id, classification, model, units_to_compare, count):
    ranks_of_units_per_image = _get_unit_activations(units_to_compare, classification, model)
    return _get_similar_images(reference_image_id, ranks_of_units_per_image, count)


def _get_similar_images(reference_image_id, features_per_image, count):
    reference_features = features_per_image[reference_image_id]

    distances = []
    for image_id in features_per_image.keys():
        if image_id == reference_image_id:
            continue
        distance = cosine_distance(reference_features, features_per_image[image_id])
        distances.append((image_id, distance))

    distances.sort(key=lambda x: x[1])
    top_image_ids = [image_id for image_id, distance in distances[:count]]
    return top_image_ids


def _get_unit_ranks(unit_ids, classification, model):
    ranks_of_units_per_image = {}

    for unit_id in unit_ids:
        for image_id, rank in _get_ranks_of_unit(unit_id, classification, model):
            if image_id in ranks_of_units_per_image:
                ranks_of_units_per_image[image_id].append(rank)
            else:
                ranks_of_units_per_image[image_id] = [rank]

    return ranks_of_units_per_image


def _get_unit_activations(unit_ids, classification, model):
    activations_of_units_per_image = {}

    for unit_id in unit_ids:
        for image_id, activation in _get_activations_of_unit(unit_id, classification, model):
            if image_id in activations_of_units_per_image:
                activations_of_units_per_image[image_id].append(activation)
            else:
                activations_of_units_per_image[image_id] = [activation]

    return activations_of_units_per_image


def _get_classification(image_id, model):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()

    select_net = "(SELECT id FROM net WHERE net = '{}')".format(model)

    select_stmt = "SELECT class_id FROM image_classification " \
                  "WHERE net_id = {net} " \
                  "AND image_id = ?;".format(net=select_net)
    c.execute(select_stmt, (image_id,))
    result = c.fetchone()[0]
    return result


def _get_all_image_ids(split='val'):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    select_stmt = "SELECT id FROM image " \
                  "WHERE split = ?"
    result = c.execute(select_stmt, (split,))
    image_ids = [row[0] for row in result]
    return image_ids


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


@lru_cache(maxsize=None)
def _get_all_annotated_units(name, model):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()

    select_net = "(SELECT id FROM net WHERE net = '{}')".format(model)
    select_doctor = "(SELECT id FROM doctor WHERE name = '{}')".format(name)

    select_stmt = "SELECT unit_id FROM unit_annotation " \
                  "WHERE net_id = {net} " \
                  "AND doctor_id = {doctor} " \
                  "AND unit_annotation.shows_concept = 1 " \
                  "ORDER BY unit_id;".format(doctor=select_doctor, net=select_net)
    result = c.execute(select_stmt)
    annotated_units = [row[0] for row in result]
    return annotated_units


@lru_cache(maxsize=None)
def _get_activations_of_unit(unit_id, classification, model):
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


@lru_cache(maxsize=None)
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
    ranks = [(row[0], 2048 - row[1]) for row in result]  # best rank is 2048 in DB
    return ranks


@lru_cache(maxsize=None)
def _get_ground_truth(image_id):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    select_stmt = "SELECT ground_truth FROM image " \
                  "WHERE id = ?;"
    c.execute(select_stmt, (image_id,))
    result = c.fetchone()[0]
    return result


@lru_cache(maxsize=None)
def _get_top_n_units(image_id, classification, model, count):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()

    select_net = "(SELECT id FROM net WHERE net = '{}')".format(model)

    select_stmt = "SELECT unit_id FROM image_unit_activation " \
                  "WHERE net_id = {net} " \
                  "AND class_id = ? " \
                  "AND image_id = ? " \
                  "AND rank > ?;".format(net=select_net)
    result = c.execute(select_stmt, (classification, image_id, 2048 - count))  # best rank is 2048 in DB
    unit_ids = [row[0] for row in result]
    return unit_ids

