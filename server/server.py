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


@app.route('/summary')
def summary():
    response_summary = backend.get_summary()
    summary2 = []
    for (name, responded_units) in response_summary:
        unquote_name = urllib.parse.unquote_plus(name)
        summary2.append((name, unquote_name, responded_units))
    return render_template('summary.html', summary=summary2)


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


@app.route('/overview/<name>/<model>/<layer>')
def overview(name, model, layer, full=False, ranked=False):
    unquote_name = urllib.parse.unquote_plus(name)
    units = backend.get_units(name, model, layer, full=full, ranked=ranked)
    num_responses = backend.get_num_responses(name)
    if ranked:
        full = False
    return render_template('overview.html', name=name, unquote_name=unquote_name, num_responses=num_responses,
                           full=full, ranked=ranked, model=model, layer=layer, units=units)


@app.route('/overview/full/<name>/<model>/<layer>')
def overview_full(name, model, layer):
    return overview(name, model, layer, full=True)


@app.route('/overview/ranked/<name>/<model>/<layer>')
def overview_ranked(name, model, layer):
    return overview(name, model, layer, full=True, ranked=True)


@app.route('/survey/<name>/<model>/<layer>/<unit>')
def survey(name, model, layer, unit, full=False, ranked=False):
    unquote_name = urllib.parse.unquote_plus(name)
    unit_id = int(unit.split("_")[1])  # looks like: unit_0076
    previous_annotations = backend.get_survey(unquote_name, model, layer, unit)
    previous_annotations = {a:a for a in previous_annotations}  # turn into dict for flask
    result = backend.get_top_images_with_activation_for_unit(unit_id, 8)
    top_images, preprocessed_top_images, activation_maps = result
    return render_template('survey.html', name=name, unquote_name=unquote_name, full=full,
                           ranked=ranked, model=model, layer=layer, unit=unit, top_images=top_images,
                           preprocessed_top_images=preprocessed_top_images, activation_maps=activation_maps, **previous_annotations)


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


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(full_path)

    return render_template('single_image.html', success=True, full_path=full_path)


@app.route('/process_image', methods=['POST'])
def process_image():
    original_path = os.path.join(app.config['UPLOAD_FOLDER'],  'test.jpg')
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'benign.jpg')
    result = backend.single_image_analysis.analyze_one_image(os.path.join('../server/static/uploads', 'benign.jpg'))

    activation_map_path = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'activation.jpg')

    top_units_and_activations = result.get_top_units(result.classification, 1)
    activation_map = top_units_and_activations[0][2]  # activation map for unit 0 => top unit

    img = Image.open(original_path)
    # normalize activation values between 0 and 255
    activation_map_normalized = backend.normalize_activation_map(activation_map)

    # resize activation map to img size
    activation_map_resized = backend.resize_activation_map(img, activation_map_normalized)

    plt.gray()  # grayscale
    plt.imsave(activation_map_path, activation_map_resized)

    activations_overlayed_path = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'benign.jpg')
    img.save(activations_overlayed_path)

    backend.grad_cam()
    return render_template('single_image.html', success=False, processed=True, full_path=processed_path,
                           top_units_and_activations=top_units_and_activations,
                           activation_map_path=activation_map_path, activations_overlayed_path=activations_overlayed_path)


@app.route('/unit_ranking_by_weights/<training_session>/<checkpoint_name>/upload')
def single_image(training_session, checkpoint_name):
    return render_template('single_image.html', success=False, processed=False, full_path='')


@app.route('/example_analysis')
def example_analysis():
    return image('cancer_05-C_0128_1.LEFT_CC.LJPEG.1.jpg')


@app.route('/image/<image_filename>')
def image(image_filename):
    image_path = os.path.join('../data/ddsm_raw/', image_filename)
    preprocessed_full_image = get_preview_of_preprocessed_image(image_path)
    preprocessed_full_image_path = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'full_image_{}.jpg'.format(uuid.uuid4()))
    preprocessed_full_image.save(preprocessed_full_image_path)
    result = backend.single_image_analysis.analyze_one_image(image_path)

    mask_dirs = ["benigns", "benign_without_callbacks", "cancers"]
    mask_path = ""
    for mask_dir in mask_dirs:
        orig_mask_path = os.path.join('../data/ddsm_masks/3class', mask_dir, image_filename[:-4] + '.png')
        if os.path.exists(orig_mask_path):
            mask_path = os.path.join(app.config['ACTIVATIONS_FOLDER'], 'mask_{}.png'.format(uuid.uuid4()))
            copyfile(orig_mask_path, mask_path)

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
