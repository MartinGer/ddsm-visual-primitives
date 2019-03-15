import os
from collections import namedtuple

Image = namedtuple("Image", "image_path ground_truth split")


def get_real_gt_per_image():
    splits = ['train', 'val', 'test']

    gt_per_image = {}

    for split in splits:
        image_list = '../data/ddsm_3class/' + split + '.txt'

        with open(image_list, 'r') as f:
            for line in f.readlines():
                patch_filename = line.strip().split(' ')[0]
                patch_gt = int(line.strip().split(' ')[1])

                full_image_name = patch_filename.split('/')[2].split('-x')[0] + '.jpg'

                if full_image_name in gt_per_image:
                    gt_per_image[full_image_name] = max(gt_per_image[full_image_name], patch_gt)
                else:
                    gt_per_image[full_image_name] = patch_gt

    return gt_per_image


def _check_that_image_table_empty(conn):
    c = conn.cursor()
    c.execute("SELECT * FROM image;")
    if c.fetchone() is not None:
        raise FileExistsError("The `image` table is already populated.")


def populate_db_with_images(conn, image_lists_path):
    _check_that_image_table_empty(conn)
    images = _get_images(image_lists_path)
    insert_statement = _generate_sql_insert(images)
    conn.execute(insert_statement)
    conn.commit()


def _get_images(image_lists_path):
    files = []

    for filename in os.listdir(image_lists_path):
        if filename.endswith(".txt"):
            files.append((filename[:-4], filename))

    images = []

    gt_per_image = get_real_gt_per_image()

    for split, filename in files:
        with open(os.path.join(image_lists_path, filename)) as file:
            for line in file.readlines():
                image_path = line.rstrip()
                ground_truth = gt_per_image[image_path]
                image = Image(image_path, ground_truth, split.rstrip())
                images.append(image)

    return images


def _generate_sql_insert(images):
    num_images = len(images)
    if num_images == 0:
        raise IndexError("No images were given")

    sql = "INSERT INTO image (image_path, ground_truth, split) VALUES "

    for i, image in enumerate(images):
        if i > 0:
            sql += ","
        sql += f" (\"{image.image_path}\", {image.ground_truth}, \"{image.split}\")"

    sql += ";"

    return sql
