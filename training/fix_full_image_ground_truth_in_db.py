import sys
from tqdm import tqdm

sys.path.insert(0, '..')
from db.database import DB


splits = ['train', 'val', 'test']

gt_per_image = {}

for split in splits:
    image_list = '../data/ddsm_3class/' + split + '.txt'

    with open(image_list, 'r') as f:
        for line in tqdm(f.readlines()):
            patch_filename = line.strip().split(' ')[0]
            patch_gt = int(line.strip().split(' ')[1])

            full_image_name = patch_filename.split('/')[2].split('-x')[0] + '.jpg'

            if full_image_name in gt_per_image:
                gt_per_image[full_image_name] = max(gt_per_image[full_image_name], patch_gt)
            else:
                gt_per_image[full_image_name] = patch_gt

db = DB()
conn = db.get_connection()

for full_image_name, ground_truth in tqdm(gt_per_image.items()):
    sql = "UPDATE image SET ground_truth = ? WHERE image_path = ?;"
    conn.execute(sql, (ground_truth, full_image_name))

conn.commit()
