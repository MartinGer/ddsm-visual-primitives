from db.database import DB


def get_class_influences_for_class(model):
    """
    Loads a checkpoint for a resnet152_3class architecture and returns last hidden layer unit to class activation
    weights. These can be used as an influence ranking of all units for each individual outcome class.

    :return: Three, one for each class outcome, rankings (sorted lists) of (unit, weight)-pairs in sorted order
    """

    class_influence_weights = {
        0: enumerate(model._modules['module'].fc.weight.data[0], 1),
        1: enumerate(model._modules['module'].fc.weight.data[1], 1),
        2: enumerate(model._modules['module'].fc.weight.data[2], 1)
    }
    class_influence_weights_sorted = {
        0: sorted(class_influence_weights[0], key=lambda x:-x[1]),
        1: sorted(class_influence_weights[1], key=lambda x:-x[1]),
        2: sorted(class_influence_weights[2], key=lambda x:-x[1])
    }

    return class_influence_weights_sorted[0], class_influence_weights_sorted[1], class_influence_weights_sorted[2]


def get_top_units_ranked():
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
    return sorted_scores

