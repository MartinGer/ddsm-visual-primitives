from flask import Flask, render_template, request, g, redirect
import urllib
import urllib.parse
import os
import sys

sys.path.insert(0, '../training')

import backend
from common import dataset
from training.unit_rankings import get_class_influences, get_top_units_ranked, cached_unit_rankings
from db.database import DB

app = Flask(__name__)


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


STATIC_DIR = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PROCESSED_FOLDER = os.path.join(STATIC_DIR, 'processed')
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
PROCESSED_FOLDER = os.path.join(STATIC_DIR, 'activation_maps')
app.config['ACTIVATIONS_FOLDER'] = PROCESSED_FOLDER

CURRENT_USER = "default"


@app.route('/')
def index(name=None):
    return render_template('login.html', name=name)


@app.route('/handle_login', methods=['POST'])
def handle_login():
    global CURRENT_USER
    name = request.form['name']
    backend.register_doctor_if_not_exists(name)
    CURRENT_USER = name
    return redirect('/home')


@app.route('/home')
def home():
    return render_template('home.html', name=CURRENT_USER)


@app.route('/checkpoints')
def checkpoints():
    db = DB()
    conn = db.get_connection()
    select_stmt = "SELECT filename FROM net"
    result = conn.execute(select_stmt)
    checkpoints = ["/".join(r[0].split("/")[-2:]) for r in result]
    return render_template('checkpoints.html',
                           checkpoints=checkpoints)


@app.route('/load_checkpoint/<training_session>/<checkpoint_name>')
def load_checkpoint(training_session, checkpoint_name):
    checkpoint_path = os.path.join('..', 'training', 'checkpoints', training_session, checkpoint_name)
    backend.init_single_image_analysis(checkpoint_path)
    return redirect('/home')


@app.route('/top_units')
def top_units():
    if cached_unit_rankings:
        return render_template('unit_ranking_by_score.html', units=cached_unit_rankings[:60])
    unit_ids = get_top_units_ranked()
    return render_template('unit_ranking_by_score.html', units=unit_ids[:60])


@app.route('/top_units_by_weights')
def unit_ranking_by_weights_for_checkpoint(unit_count=20, patch_count=6):
    if not backend.single_image_analysis:
        return redirect('/checkpoints')
    model = backend.single_image_analysis.get_model()
    sorted_influences = get_class_influences(model)
    top_patches = {}
    appearances_in_top_units = {}

    for class_id, count in ((0, 4), (1, unit_count), (2, unit_count)):
        for unit_id, influence in sorted_influences[class_id][:count]:
            top_patches[unit_id] = backend.get_top_patches_for_unit(unit_id, patch_count)
            appearances_in_top_units[unit_id] = backend.get_appearances_in_top_units(unit_id, class_id)

    return render_template('unit_ranking_by_weights_for_checkpoint.html',
                           sorted_weights_class_0=sorted_influences[0][:4],
                           sorted_weights_class_1=sorted_influences[1][:unit_count],
                           sorted_weights_class_2=sorted_influences[2][:unit_count],
                           top_patches=top_patches,
                           appearances_in_top_units=appearances_in_top_units)


@app.route('/unit/<unit_id>')
def unit(unit_id):
    if not backend.single_image_analysis:
        return redirect('/checkpoints')
    result = backend.get_top_images_and_heatmaps_for_unit(unit_id, 4)
    top_images, preprocessed_top_images, heatmaps = result
    top_patches, patch_heatmaps = backend.get_top_patches_and_heatmaps_for_unit(unit_id, 8)
    return render_template('unit.html',
                           unit_id=unit_id,
                           top_images=top_images,
                           preprocessed_top_images=preprocessed_top_images,
                           heatmaps=heatmaps,
                           top_patches=top_patches,
                           patch_heatmaps=patch_heatmaps)


@app.route('/image/<image_filename>')
def image(image_filename):
    if not backend.single_image_analysis:
        return redirect('/checkpoints')
    image_path = os.path.join('../data/ddsm_raw/', image_filename)
    image_name = image_filename[:-4]
    preprocessed_full_image_path = backend.get_preprocessed_image_path(image_filename)
    preprocessed_mask_path = backend.get_preprocessed_mask_path(image_filename)
    preprocessing_descr = dataset.preprocessing_description()

    result = backend.single_image_analysis.analyze_one_image(image_path)

    units_to_show = 10
    top_units_and_activations = result.get_top_units(result.classification, units_to_show)
    heatmap_paths = backend.get_heatmap_paths_for_top_units(image_filename, top_units_and_activations, units_to_show)

    return render_template('image.html',
                           image_path=result.image_path,
                           image_name=image_name,
                           preprocessed_full_image_path=preprocessed_full_image_path,
                           preprocessed_mask_path=preprocessed_mask_path,
                           checkpoint_path=result.checkpoint_path,
                           preprocessing_descr=preprocessing_descr,
                           classification=result.classification,
                           class_probs=result.class_probs,
                           top_units_and_activations=top_units_and_activations,
                           heatmap_paths=heatmap_paths)


@app.route('/survey/<name>/<model>/<unit>')
def survey(name, model, unit):
    unquote_name = urllib.parse.unquote_plus(name)
    unit_id = int(unit.split("_")[1])  # looks like: unit_0076
    previous_survey = backend.get_survey(unquote_name, model, unit)
    if previous_survey:
        shows_phenomena, description = previous_survey
        if shows_phenomena:
            shows_phenomena = 'true'
            previous_annotations = {a: a for a in description}  # turn into dict for flask
        else:
            shows_phenomena = 'false'
            previous_annotations = {}
    else:
        shows_phenomena = 'true'
        previous_annotations = {}
    result = backend.get_top_images_and_heatmaps_for_unit(unit_id, 8)
    top_images, preprocessed_top_images, activation_maps = result
    return render_template('survey.html', name=name, unquote_name=unquote_name, model=model, unit=unit,
                           top_images=top_images, preprocessed_top_images=preprocessed_top_images, activation_maps=activation_maps,
                           shows_phenomena=shows_phenomena, **previous_annotations)


@app.route('/handle_survey', methods=['POST'])
def handle_survey():
    name = urllib.parse.unquote_plus(request.form['name'])  # doctor username
    model = request.form['model']  # resnet152
    unit = request.form['unit']   # unit_0076
    shows_phenomena = request.form['shows_phenomena']
    phenomena = [p for p in request.form if p.startswith('phe')]
    backend.store_survey(name, model, unit, shows_phenomena, phenomena)
    return redirect('/home/{}'.format(name))


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(full_path)

    return render_template('single_image.html', success=True, full_path=full_path, image_filename=file.filename)


@app.route('/unit_ranking_by_weights/<training_session>/<checkpoint_name>/upload')
def single_image(training_session, checkpoint_name):
    return render_template('single_image.html', success=False, processed=False)


@app.route('/_image/<image_filename>')
def _image(image_filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

    preprocessed_full_image_path = backend.get_preprocessed_image_path(image_filename, app.config['UPLOAD_FOLDER'])
    preprocessed_mask_path = ""  # no mask available for new images
    preprocessing_descr = dataset.preprocessing_description()

    result = backend.single_image_analysis.analyze_one_image(image_path)

    units_to_show = 10
    top_units_and_activations = result.get_top_units(result.classification, units_to_show)
    heatmap_paths = backend.get_heatmap_paths_for_top_units(image_filename, top_units_and_activations, units_to_show, app.config['UPLOAD_FOLDER'])

    return render_template('image.html',
                           image_path=result.image_path,
                           preprocessed_full_image_path=preprocessed_full_image_path,
                           preprocessed_mask_path=preprocessed_mask_path,
                           checkpoint_path=result.checkpoint_path,
                           preprocessing_descr=preprocessing_descr,
                           classification=result.classification,
                           class_probs=result.class_probs,
                           top_units_and_activations=top_units_and_activations,
                           heatmap_paths=heatmap_paths)


@app.route('/example_analysis')
def example_analysis():
    # good examples:
    # cancer_15-B_3504_1.RIGHT_CC.LJPEG.1.jpg -> 99% cancer, two spots
    return image('cancer_09-B_3134_1.RIGHT_CC.LJPEG.1.jpg')


