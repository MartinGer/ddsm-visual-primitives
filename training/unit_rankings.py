from db.database import DB


_cached_unit_rankings_by_weights = None


def get_top_units_by_class_influences(count):
    """
    Returns last hidden layer unit to class activation weights from database.
    These can be used as an influence ranking of all units for each individual outcome class.

    :return: Three, one for each class outcome, rankings (sorted lists) of (unit, weight)-pairs in sorted order
    """
    global _cached_unit_rankings_by_weights
    if _cached_unit_rankings_by_weights:
        return _cached_unit_rankings_by_weights

    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    class_influences = []
    for class_id in range(3):
        select_stmt = "SELECT unit_id, influence, appearances_in_top_units FROM unit_class_influence " \
                      "WHERE class_id = ? ORDER BY influence DESC " \
                      "LIMIT ?;"
        c.execute(select_stmt, (class_id, count))
        result = [(row[0], row[1], row[2]) for row in c.fetchall()]
        class_influences.append(result)

    _cached_unit_rankings_by_weights = class_influences
    return class_influences


_cached_unit_rankings_by_appearances_in_top_units = None


def get_top_units_by_appearances_in_top_units(count):
    global _cached_unit_rankings_by_appearances_in_top_units
    if _cached_unit_rankings_by_appearances_in_top_units:
        return _cached_unit_rankings_by_appearances_in_top_units

    db = DB()
    conn = db.get_connection()
    c = conn.cursor()
    ranked_units = []
    for class_id in range(3):
        select_stmt = "SELECT unit_id, influence, appearances_in_top_units FROM unit_class_influence " \
                      "WHERE class_id = ? ORDER BY appearances_in_top_units DESC " \
                      "LIMIT ?;"
        c.execute(select_stmt, (class_id, count))
        result = [(row[0], row[1], row[2]) for row in c.fetchall()]
        ranked_units.append(result)

    _cached_unit_rankings_by_appearances_in_top_units = ranked_units
    return ranked_units


_cached_unit_rankings_custom_score = None


def get_top_units_ranked():
    global _cached_unit_rankings_custom_score
    if _cached_unit_rankings_custom_score:
        return _cached_unit_rankings_custom_score

    db = DB()
    conn = db.get_connection()
    c = conn.cursor()

    # image id, ground_truth class, min_act(all classes), max_act(all_classes)
    select_stmt = "SELECT image_id, image.ground_truth, MIN(activation), MAX(activation) " \
                  "FROM image_unit_activation INNER JOIN image ON image_unit_activation.image_id = image.id " \
                  "WHERE image.split='val' GROUP BY image_id;"
    result = c.execute(select_stmt)
    rows = [r for r in result]
    if not rows:
        raise AssertionError('Could not retrieve information from db')

    image_info = {tup[0] : tup[1:] for tup in rows}

    # unit_id, image_id, class_id, activation
    select_stmt = "SELECT unit_id, image_id, class_id, activation FROM image_unit_activation INNER JOIN image ON image_unit_activation.image_id = image.id WHERE image.split='val';"

    result = c.execute(select_stmt)
    rows = [r for r in result]
    if not rows:
        raise AssertionError('Could not retrieve information from db')

    scores = dict()
    c = 0
    for unit_info in rows:
        if c % 100000 == 0:
            print(str(c) + "/" + str(len(rows)))
        c += 1

        unit_id, image_id, class_id, activation = unit_info
        ground_truth, min_act, max_act = image_info[image_id]

        bucket_size = (max_act - min_act) / 3
        if class_id == 0:  # does not make sense to take normal class into consideration
            continue
        if class_id == ground_truth:  # given activation value is for the correct class
            if activation <= min_act + bucket_size:  # low activation value
                scores[unit_id] = scores.get(unit_id, 0) + 1
            elif activation <= min_act + bucket_size*2:  # medium activation value
                scores[unit_id] = scores.get(unit_id, 0) + 3
            else:  # high activation value
                scores[unit_id] = scores.get(unit_id, 0) + 5
        else:  # given activation value is for wrong class
            if activation <= min_act + bucket_size:  # low activation value
                scores[unit_id] = scores.get(unit_id, 0) - 1
            elif activation <= min_act + bucket_size*2:  # medium activation value
                scores[unit_id] = scores.get(unit_id, 0) - 3
            else:  # high activation value
                scores[unit_id] = scores.get(unit_id, 0) - 5

    sorted_scores = sorted(scores, key=scores.get, reverse=True)  # list of unit_ids
    _cached_unit_rankings_custom_score = sorted_scores
    return sorted_scores

