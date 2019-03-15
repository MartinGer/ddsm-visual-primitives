import sys
import os
from functools import lru_cache
from multiprocessing import Pool
import time
import random
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, '..')
from db.database import DB
from training.common import dataset


METRIC_CALCULATION_RUNNING = False

patchname_to_gt = None


def _get_all_patches_in_db():
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    select_stmt = "SELECT DISTINCT patch_filename FROM patch_unit_activation"
    result = c.execute(select_stmt, ())
    patch_names = [row[0] for row in result]
    return patch_names


def _get_patches(split='val'):
    patches_in_db = _get_all_patches_in_db()
    with open(os.path.join('..', 'data', 'ddsm_labels', '3class', split+'.txt'), 'r') as f:
        lines = f.readlines()
        sanitized_lines = [l.strip() for l in lines]
        split_lines = [l.split(' ') for l in sanitized_lines]
        # only 100 out of all normal patches are in DB, filter the missing ones out:
        filtered_patches = [(name, gt) for name, gt in split_lines if name in patches_in_db]
        global patchname_to_gt
        patchname_to_gt = dict(filtered_patches)
    return filtered_patches


@lru_cache(maxsize=None)
def _get_balanced_list_of_patchnames(patch_count=10):
    balanced = defaultdict(list)
    for patch in _get_patches():
        if len(balanced[patch[1]]) != patch_count:
            balanced[patch[1]].append(patch[0])
        if len(balanced['0']) == patch_count and len(balanced['1']) == patch_count and len(balanced['2']) == patch_count:
            break
    return balanced['0'] + balanced['1'] + balanced['2']


def _get_ground_truth(patchname):
    global patchname_to_gt
    return min(int(patchname_to_gt[patchname]), 1)  # merge benign and malignant to one category


def print_all_similarity_scores(name, model):
    global METRIC_CALCULATION_RUNNING
    if METRIC_CALCULATION_RUNNING:
        return
    METRIC_CALCULATION_RUNNING = True

    print("print_all_similarity_scores_patches started")

    for units_to_compare in ('top_10_units', 'top_annotated', 'all_annotated'):
        for feature_to_compare in ('activation', 'rank'):
            _print_average_similarity_score(name, model, units_to_compare, feature_to_compare, patch_count=20)

    METRIC_CALCULATION_RUNNING = False


def _print_average_similarity_score(name, model, units_to_compare, feature_to_compare, patch_count):
    print("_get_average_similarity_score started:", units_to_compare, feature_to_compare, patch_count)
    balanced_patches = _get_balanced_list_of_patchnames(100)
    similar_patches_with_same_gt = 0

    for patchname in tqdm(balanced_patches):
        gt = _get_ground_truth(patchname)
        gt_distribution = _get_similarity_score_by_patchname(patchname, name, model, units_to_compare, feature_to_compare, patch_count)[0]
        similar_patches_with_same_gt += gt_distribution[gt]

    avg_imgs_with_same_gt = similar_patches_with_same_gt / len(balanced_patches)
    print("Avg. similar patches with same ground truth: {:.2f} of {} -> {:.2f}%".format(avg_imgs_with_same_gt, patch_count, (avg_imgs_with_same_gt / patch_count) * 100))


def _get_classification(patchname, model):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()

    select_net = "(SELECT id FROM net WHERE net = '{}')".format(model)

    select_stmt = "SELECT class_id FROM patch_classification " \
                  "WHERE net_id = {net} " \
                  "AND patch_filename = ?;".format(net=select_net)
    c.execute(select_stmt, (patchname,))
    result = c.fetchone()[0]
    return min(result, 1)  # merge benign and malignant to one category


@lru_cache(maxsize=None)
def _get_top_n_units(patchname, classification, model, count):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()

    select_net = "(SELECT id FROM net WHERE net = '{}')".format(model)

    select_stmt = "SELECT unit_id FROM patch_unit_activation " \
                  "WHERE net_id = {net} " \
                  "AND class_id = ? " \
                  "AND patch_filename = ? " \
                  "AND rank > ?;".format(net=select_net)
    result = c.execute(select_stmt, (classification, patchname, 2048 - count))  # best rank is 2048 in DB
    unit_ids = [row[0] for row in result]
    return unit_ids


def _get_annotated_top_units(patchname, classification, name, model):
    annotated_units = _get_all_annotated_units(name, model)
    ranks_of_units_per_patch = _get_unit_ranks(annotated_units, classification, model)
    annotated_top_units = [annotated_units[i] for i in range(len(annotated_units)) if
                           ranks_of_units_per_patch[patchname][i] < 10]
    return annotated_top_units


def _get_unit_ranks(unit_ids, classification, model):
    ranks_of_units_per_patch = {}

    for unit_id in unit_ids:
        for patchname, rank in _get_ranks_of_unit(unit_id, classification, model):
            if patchname in ranks_of_units_per_patch:
                ranks_of_units_per_patch[patchname].append(rank)
            else:
                ranks_of_units_per_patch[patchname] = [rank]

    return ranks_of_units_per_patch


@lru_cache(maxsize=None)
def _get_ranks_of_unit(unit_id, classification, model):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()

    select_net = "(SELECT id FROM net WHERE net = '{}')".format(model)

    select_stmt = "SELECT patch_filename, rank FROM patch_unit_activation " \
                  "WHERE net_id = {net} " \
                  "AND class_id = ? " \
                  "AND unit_id = ?;".format(net=select_net)
    result = c.execute(select_stmt, (classification, unit_id))
    ranks = [(row[0], 2048 - row[1]) for row in result]  # best rank is 2048 in DB
    return ranks


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


def _get_similar_patches_by_activations(reference_patchname, classification, model, units_to_compare, count):
    ranks_of_units_per_patch = _get_unit_activations(units_to_compare, classification, model)
    return _get_similar_patches(reference_patchname, ranks_of_units_per_patch, count)


def _get_unit_activations(unit_ids, classification, model):
    activations_of_units_per_patch = {}

    for unit_id in unit_ids:
        for patchname, activation in _get_activations_of_unit(unit_id, classification, model):
            if patchname in activations_of_units_per_patch:
                activations_of_units_per_patch[patchname].append(activation)
            else:
                activations_of_units_per_patch[patchname] = [activation]

    return activations_of_units_per_patch


def _get_similar_patches(reference_patchname, features_per_patch, count):
    reference_features = features_per_patch[reference_patchname]

    distances = []
    for patchname in features_per_patch.keys():
        if patchname == reference_patchname:
            continue
        distance = cosine_distance(reference_features, features_per_patch[patchname])
        distances.append((patchname, distance))

    distances.sort(key=lambda x: x[1])
    top_patchnames = [patchname for patchname, distance in distances[:count]]
    return top_patchnames


@lru_cache(maxsize=None)
def _get_activations_of_unit(unit_id, classification, model):
    db = DB()
    conn = db.get_connection()
    c = conn.cursor()

    select_net = "(SELECT id FROM net WHERE net = '{}')".format(model)

    select_stmt = "SELECT patch_filename, activation FROM patch_unit_activation " \
                  "WHERE net_id = {net} " \
                  "AND class_id = ? " \
                  "AND unit_id = ?;".format(net=select_net)
    result = c.execute(select_stmt, (classification, unit_id))
    activations = [(row[0], row[1]) for row in result]
    return activations


def _get_similar_patches_by_ranks(reference_patchname, classification, model, units_to_compare, count):
    ranks_of_units_per_patch = _get_unit_ranks(units_to_compare, classification, model)
    return _get_similar_patches(reference_patchname, ranks_of_units_per_patch, count)


def _get_similarity_score_by_patchname(reference_patchname, name, model, units_to_compare='top_10_units', feature_to_compare='activation', patch_count=20):
    classification = _get_classification(reference_patchname, model)

    if units_to_compare == 'top_10_units':
        _units_to_compare = _get_top_n_units(reference_patchname, classification, model, 10)
    elif units_to_compare == 'top_annotated':
        _units_to_compare = _get_annotated_top_units(reference_patchname, classification, name, model)
    elif units_to_compare == 'all_annotated':
        _units_to_compare = _get_all_annotated_units(name, model)
    else:
        raise ValueError("Unknown content of units_to_compare.")

    if not _units_to_compare:
        print("No units to compare for this patch -> can't find similar patches. (This can happen for normal patches.)")
        return (0, 0, 0), [], []

    if feature_to_compare == 'activation':
        top_patchnames = _get_similar_patches_by_activations(reference_patchname, classification, model, _units_to_compare, patch_count)
    elif feature_to_compare == 'rank':
        top_patchnames = _get_similar_patches_by_ranks(reference_patchname, classification, model, _units_to_compare, patch_count)
    else:
        raise ValueError("Unknown content of feature_to_compare.")

    ground_truth_of_top_patches = np.asarray([_get_ground_truth(patchname) for patchname in top_patchnames])
    gt_distribution = ((ground_truth_of_top_patches == 0).sum(), (ground_truth_of_top_patches == 1).sum(), (ground_truth_of_top_patches == 2).sum())

    return gt_distribution, top_patchnames, ground_truth_of_top_patches


if __name__ == '__main__':
    print_all_similarity_scores('Prof Dr Bick', 'resnet152')

