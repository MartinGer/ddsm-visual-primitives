import sys
import os
from functools import lru_cache
from multiprocessing import Pool
import time

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, '..')
from db.database import DB
from training.common import dataset


METRIC_CALCULATION_RUNNING = False


def similarity_metric(image_filename, name, model):
    reference_image_id = _get_image_id(image_filename)
    gt_distribution, top20_image_ids, ground_truth_of_top20 = _get_similarity_score_by_image_id(reference_image_id, name, model)
    top20_image_paths = [_get_image_path(image_id) for image_id in top20_image_ids]
    return gt_distribution, top20_image_paths, ground_truth_of_top20


def similarity_metric_for_uploaded_image(findings, analysis_result, model):
    top_units_and_activations = analysis_result.get_top_units(analysis_result.classification, 10)
    findings_ids = np.array(findings).transpose()[0]

    annotated_top_units = []
    for findings_id in findings_ids:
        for top_unit in top_units_and_activations:
            if int(findings_id) == int(top_unit[0]):
                annotated_top_units.append(int(findings_id))

    ranks_of_units_per_image = {}
    for unit_id in set(annotated_top_units):
        for image_id, rank in _get_ranks_of_unit(unit_id, analysis_result.classification, model):
            if image_id in ranks_of_units_per_image:
                ranks_of_units_per_image[image_id].append(rank)
            else:
                ranks_of_units_per_image[image_id] = [rank]

    reference_ranks = []
    all_units_and_activations = analysis_result.get_top_units(analysis_result.classification, 2048)
    for unit_idx in range(len(all_units_and_activations)):
        for annotated_unit in set(annotated_top_units):
            if all_units_and_activations[unit_idx][0] == annotated_unit:
                reference_ranks.append(unit_idx)     # rank of annotated units for the uploaded image

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


def print_all_similarity_scores(name, model):
    global METRIC_CALCULATION_RUNNING
    if METRIC_CALCULATION_RUNNING:
        return
    METRIC_CALCULATION_RUNNING = True

    print("print_all_similarity_scores started")

    for units_to_compare in ('all_annotated', 'top_10_units', 'top_annotated'):
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
    print("Avg. similar images with same ground truth: {:.2f} of {} -> {:.2f}%".format(avg_imgs_with_same_gt, image_count, (avg_imgs_with_same_gt / image_count) * 100))


def _get_similarity_score_by_image_id(reference_image_id, name, model, units_to_compare='top_annotated', feature_to_compare='activation', image_count=20):
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
        # print("No units to compare for this image -> can't find similar images.")
        return (0, 0, 0), [], []

    if feature_to_compare == 'activation':
        top_image_ids = _get_similar_images_by_activations(reference_image_id, classification, model, _units_to_compare, image_count)
    elif feature_to_compare == 'rank':
        top_image_ids = _get_similar_images_by_ranks(reference_image_id, classification, model, _units_to_compare, image_count)
    else:
        raise ValueError("Unknown content of feature_to_compare.")

    ground_truth_of_top_images = np.asarray([_get_classification(image_id, model) for image_id in top_image_ids])
    gt_distribution = ((ground_truth_of_top_images == 0).sum(), (ground_truth_of_top_images == 1).sum(), (ground_truth_of_top_images == 2).sum())

    return gt_distribution, top_image_ids, ground_truth_of_top_images


def _get_ssim_similarity_score(image_count=20):
    image_names = _get_all_image_names()

    similar_imgs_with_same_gt = 0
    pool = Pool()

    for i, reference_image_name in enumerate(image_names):
        print("Finding similar images using SSIM for image {} of {}".format(i, len(image_names)))
        begin = time.time()
        reference_image = _get_preprocessed_image_as_array(os.path.join('../data/ddsm_raw', reference_image_name))

        # TODO: exclude MLO AND CC images?
        similarites = pool.map(_get_similarity, [(reference_image, image_name) for image_name in image_names if image_name != reference_image_name])

        similarites.sort(key=lambda x: x[1], reverse=True)
        top_image_names = [image_path for image_path, similarity in similarites[:image_count]]

        ground_truth_of_top_images = np.asarray([_get_ground_truth_by_name(image_name) for image_name in top_image_names])
        gt_distribution = ((ground_truth_of_top_images == 0).sum(), (ground_truth_of_top_images == 1).sum(), (ground_truth_of_top_images == 2).sum())
        gt = _get_ground_truth_by_name(reference_image_name)
        similar_imgs_with_same_gt += gt_distribution[gt]
        print("GT distribution:", gt_distribution, gt, (similar_imgs_with_same_gt / (i + 1)))
        end = time.time()
        print("Estimated Remaining Time: {:.1f}h".format(((end - begin) * (len(image_names) - i)) / 3600))

    avg_imgs_with_same_gt = similar_imgs_with_same_gt / len(image_names)
    print("Avg. similar images with same ground truth (SSIM): {:.2f} of {} -> {:.2f}%".format(avg_imgs_with_same_gt, image_count, (avg_imgs_with_same_gt / image_count) * 100))


def _get_similarity(input):
    reference_image, image_name = input
    img = _get_preprocessed_image_as_array(os.path.join('../data/ddsm_raw', image_name))
    return (image_name, ssim(reference_image, img))


@lru_cache(maxsize=None)
def _get_preprocessed_image_as_array(path):
    image = Image.open(path)
    image = dataset.resize_and_pad_image(image, dataset.IMAGE_SIZE_TO_ANALYZE, dataset.TARGET_ASPECT_RATIO, augmentation=False)
    image = dataset.remove_background_noise(image, augmentation=False)
    return image


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
    image_ids = []
    for class_id in (0, 1, 2):
        select_stmt = "SELECT id FROM image " \
                      "WHERE split = ? " \
                      "AND ground_truth = ? " \
                      "ORDER BY id;"
                      # "LIMIT (" \
                      # "  SELECT MIN(images_per_class) " \
                      # "  FROM (" \
                      # "    SELECT COUNT(id) images_per_class " \
                      # "    FROM image " \
                      # "    WHERE split = ? " \
                      # "    GROUP BY ground_truth" \
                      # "  )" \
                      # ")"
        result = c.execute(select_stmt, (split, class_id))
        image_ids += [row[0] for row in result]
    return image_ids


def _get_all_image_names(split='val'):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    image_names = []
    for class_id in (0, 1, 2):
        select_stmt = "SELECT image_path, id FROM image " \
                      "WHERE split = ? " \
                      "AND ground_truth = ? " \
                      "ORDER BY id;"
                      # "LIMIT (" \
                      # "  SELECT MIN(images_per_class) " \
                      # "  FROM (" \
                      # "    SELECT COUNT(id) images_per_class " \
                      # "    FROM image " \
                      # "    WHERE split = ? " \
                      # "    GROUP BY ground_truth" \
                      # "  )" \
                      # ")"
        result = c.execute(select_stmt, (split, class_id))
        image_names += [row[0] for row in result]
    return image_names


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
def _get_ground_truth_by_name(image_name):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    select_stmt = "SELECT ground_truth FROM image " \
                  "WHERE image_path = ?;"
    c.execute(select_stmt, (image_name,))
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


if __name__ == '__main__':
    print_all_similarity_scores('Prof Dr Bick', 'resnet152')
    #_get_ssim_similarity_score()
