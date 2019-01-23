from flask import Flask, render_template, request
from flask import redirect
import urllib
import urllib.parse
import os
import backend
import sys
from PIL import Image
import matplotlib.pyplot as plt
import uuid
from shutil import copyfile

sys.path.insert(0, '../training')

from common.dataset import get_preview_of_preprocessed_image, preprocessing_description
from training.unit_rankings import get_class_influences_for_class
from PIL import ImageOps

app = Flask(__name__)

STATIC_DIR = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PROCESSED_FOLDER = os.path.join(STATIC_DIR, 'processed')
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
PROCESSED_FOLDER = os.path.join(STATIC_DIR, 'activation_maps')
app.config['ACTIVATIONS_FOLDER'] = PROCESSED_FOLDER


@app.route('/')
def index(name=None):
    return render_template('index.html', name=name)


@app.route('/handle_login', methods=['POST'])
def handle_login():
    name = request.form['name']
    backend.register_doctor_if_not_exists(name)
    return redirect('/home/{}'.format(name))


@app.route('/home/<name>')
def home(name):
    unquote_name = urllib.parse.unquote_plus(name)
    responded_units = backend.get_responded_units(name)
    return render_template('home.html', name=name, unquote_name=unquote_name, responded_units=responded_units)


######################################################################################TODO: fix survey routes


@app.route('/survey/<name>/<model>/<layer>/<unit>')
def survey(name, model, layer, unit, full=False, ranked=False):
    unquote_name = urllib.parse.unquote_plus(name)
    unit_id = int(unit.split("_")[1])  # looks like: unit_0076
    previous_survey = backend.get_survey(unquote_name, model, layer, unit)
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
    result = backend.get_top_images_with_activation_for_unit(unit_id, 8)
    top_images, preprocessed_top_images, activation_maps = result
    return render_template('survey.html', name=name, unquote_name=unquote_name, full=full,
                           ranked=ranked, model=model, layer=layer, unit=unit, top_images=top_images,
                           preprocessed_top_images=preprocessed_top_images, activation_maps=activation_maps,
                           shows_phenomena=shows_phenomena, **previous_annotations)


@app.route('/survey/full/<name>/<model>/<layer>/<unit>')
def survey_full(name, model, layer, unit):
    return survey(name, model, layer, unit, full=True)


@app.route('/survey/ranked/<name>/<model>/<layer>/<unit>')
def survey_ranked(name, model, layer, unit):
    return survey(name, model, layer, unit, ranked=True)


@app.route('/handle_survey', methods=['POST'])
def handle_survey(full=False, ranked=False):
    name = urllib.parse.unquote_plus(request.form['name'])  # doctor username
    model = request.form['model']  # resnet152
    layer = request.form['layer']  # layer4
    unit = request.form['unit']   # unit_0076
    shows_phenomena = request.form['shows_phenomena']
    phenomena = [p for p in request.form if p.startswith('phe')]
    backend.store_survey(name, model, layer, unit, shows_phenomena, phenomena)
    if ranked:
        return redirect('/overview/ranked/{}/{}/{}#{}'.format(name, model, layer, unit))
    elif full:
        return redirect('/overview/full/{}/{}/{}#{}'.format(name, model, layer, unit))
    else:
        return redirect('/overview/{}/{}/{}#{}'.format(name, model, layer, unit))


@app.route('/handle_survey/full', methods=['POST'])
def handle_survey_full():
    return handle_survey(full=True)


@app.route('/handle_survey/ranked', methods=['POST'])
def handle_survey_ranked():
    return handle_survey(ranked=True)

######################################################################################


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(full_path)
    return render_template('single_image.html', success=True, full_path=full_path, image_filename=file.filename)


@app.route('/unit_ranking_by_weights/<training_session>/<checkpoint_name>/upload')
def single_image(training_session, checkpoint_name):
    return render_template('single_image.html', success=False, processed=False)


@app.route('/image/<image_filename>')
def image(image_filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    preprocessed_full_image = get_preview_of_preprocessed_image(image_path)
    preprocessed_full_image_path = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'full_image_{}.jpg'.format(uuid.uuid4()))
    preprocessed_full_image.save(preprocessed_full_image_path)
    result = backend.single_image_analysis.analyze_one_image(image_path)

    mask_dirs = ["benigns", "benign_without_callbacks", "cancers"]
    mask_path = ""
    for mask_dir in mask_dirs:
        orig_mask_path = os.path.join('../data/ddsm_masks/3class', mask_dir, image_filename[:-4] + '.png')
        if os.path.exists(orig_mask_path):
            mask_preprocessed = get_preview_of_preprocessed_image(orig_mask_path)
            mask_preprocessed = ImageOps.colorize(ImageOps.equalize(mask_preprocessed), (0, 0, 0), (255, 0, 0))
            mask_path = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'mask_{}.png'.format(uuid.uuid4()))
            mask_preprocessed.save(mask_path)

    units_to_show = 10
    top_units_and_activations = result.get_top_units(result.classification, units_to_show)

    activation_maps = []

    for i in range(units_to_show):
        activation_map = top_units_and_activations[i][2]  # activation map for unit with rank i
        act_map_img = backend.to_heatmap(activation_map)
        act_map_img = act_map_img.resize(preprocessed_full_image.size, resample=Image.BICUBIC)

        activation_map_path = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'activation_{}.jpg'.format(uuid.uuid4()))
        act_map_img.save(activation_map_path, "JPEG")
        activation_maps.append(activation_map_path)

    preprocessing_descr = preprocessing_description()

    return render_template('image.html',
                           image_path=result.image_path,
                           preprocessed_full_image_path=preprocessed_full_image_path,
                           mask_path=mask_path,
                           checkpoint_path=result.checkpoint_path,
                           preprocessing_descr=preprocessing_descr,
                           classification=result.classification,
                           class_probs=result.class_probs,
                           top_units_and_activations=top_units_and_activations,
                           activation_maps=activation_maps)


@app.route('/unit/<unit_id>')
def unit(unit_id):
    result = backend.get_top_images_with_activation_for_unit(unit_id, 4)
    top_images, preprocessed_top_images, activation_maps = result
    return render_template('unit.html',
                           unit_id=unit_id,
                           top_images=top_images,
                           preprocessed_top_images=preprocessed_top_images,
                           activation_maps=activation_maps)


@app.route('/unit_ranking_by_weights')
def unit_ranking_by_weights():
    sessions = sorted(os.listdir(os.path.join('..', 'training', 'checkpoints_full_images')))
    return render_template('unit_ranking_by_weights.html',
                           session=False,
                           links=sessions)


@app.route('/unit_ranking_by_weights/<training_session>')
def unit_ranking_by_weights_for_session(training_session):
    checkpoints = sorted(os.listdir(os.path.join('..', 'training', 'checkpoints_full_images', training_session)))
    return render_template('unit_ranking_by_weights.html',
                           session=training_session,
                           links=checkpoints)


@app.route('/unit_ranking_by_weights/<training_session>/<checkpoint_name>')
def unit_ranking_by_weights_for_checkpoint(training_session, checkpoint_name):
    checkpoint_path = os.path.join('..', 'training', 'checkpoints_full_images', training_session, checkpoint_name)
    sorted_weights_class_0, sorted_weights_class_1, sorted_weights_class_2 = get_class_influences_for_class(checkpoint_path)
    return render_template('unit_ranking_by_weights_for_checkpoint.html',
                           session=training_session,
                           link=checkpoint_name,
                           sorted_weights_class_0=sorted_weights_class_0,
                           sorted_weights_class_1=sorted_weights_class_1,
                           sorted_weights_class_2=sorted_weights_class_2)
