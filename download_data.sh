cd data
if [type wget > /dev/null]; then
  wget http://data.csail.mit.edu/places/medical/data/ddsm_patches.tar.gz
  wget http://data.csail.mit.edu/places/medical/data/ddsm_labels.tar.gz
  wget http://data.csail.mit.edu/places/medical/data/ddsm_raw.tar.gz
  wget http://data.csail.mit.edu/places/medical/data/ddsm_raw_image_lists.tar.gz
  wget http://data.csail.mit.edu/places/medical/data/ddsm_masks.tar.gz
else
  curl -O http://data.csail.mit.edu/places/medical/data/ddsm_patches.tar.gz
  curl -O http://data.csail.mit.edu/places/medical/data/ddsm_labels.tar.gz
  curl -O http://data.csail.mit.edu/places/medical/data/ddsm_raw.tar.gz
  curl -O http://data.csail.mit.edu/places/medical/data/ddsm_raw_image_lists.tar.gz
  curl -O http://data.csail.mit.edu/places/medical/data/ddsm_masks.tar.gz
fi
tar -xf ddsm_patches.tar.gz
tar -xf ddsm_labels.tar.gz
tar -xf ddsm_raw.tar.gz
tar -xf ddsm_raw_image_lists.tar.gz
tar -xf ddsm_masks.tar.gz
