import pickle
import json
import os
import struct
import sys
import glob
from tesserocr import PyTessBaseAPI, RIL
import PIL
import cv2
from os.path import basename, join
from multiprocessing import Pool, cpu_count
import logging
import re
import subprocess
import numpy as np
from functools import partial
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from math import ceil
from sklearn.feature_extraction.image import extract_patches
import time
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


def create_hog():
    desc = cv2.HOGDescriptor((20, 20), (10, 10), (5, 5), (5, 5), 9)
    def compute(arr):
        return desc.compute(arr)
    return compute

hog = create_hog()


def load_annots(path):
    return [x for x in json.load(open(path)) if len(x['annotations']) > 0]


def format_line(d):
    coords = '  '.join(['{} {} {} {}'.format(int(x['x'] if x['x'] > 0 else 0), int(x['y'] if x['y'] > 0  else 0),
                                             int(x['width']), int(x['height'])) for x in d['annotations']])
    return '{} {} {}'.format(d['filename'], len(d['annotations']), coords)


def write_annotations(in_path, out_path):
    annot_lines = [format_line(x) + '\n' for x in load_annots(in_path)]
    with open(out_path, 'w') as of:
        of.writelines(annot_lines)


def merge_vec_files(vec_directory, output_vec_file):
    """
    Iterates throught the .vec files in a directory and combines them.

    (1) Iterates through files getting a count of the total images in the .vec files
    (2) checks that the image sizes in all files are the same

    The format of a .vec file is:

    4 bytes denoting number of total images (int)
    4 bytes denoting size of images (int)
    2 bytes denoting min value (short)
    2 bytes denoting max value (short)

    ex: 	6400 0000 4605 0000 0000 0000

        hex		6400 0000  	4605 0000 		0000 		0000
                # images  	size of h * w		min		max
        dec	    	100     	1350			0 		0

    :type vec_directory: string
    :param vec_directory: Name of the directory containing .vec files to be combined.
                Do not end with slash. Ex: '/Users/username/Documents/vec_files'

    :type output_vec_file: string
    :param output_vec_file: Name of aggregate .vec file for output.
        Ex: '/Users/username/Documents/aggregate_vec_file.vec'

    """

    # Check that the .vec directory does not end in '/' and if it does, remove it.
    files = glob.glob('{0}/*.vec'.format(vec_directory))

    prev_image_size = 0
    try:
        with open(files[0], 'rb') as vecfile:
            content = ''.join(str(line) for line in vecfile.readlines())
            val = struct.unpack('<iihh', content[:12])
            prev_image_size = val[1]
    except IOError as e:
        logging.error('An IO error occured while processing the file: {0}'.format(files[0]))

    # Get the total number of images
    total_num_images = 0
    num_files = len(files)
    for f in files:
        try:
            with open(f, 'rb') as vecfile:
                content = ''.join(str(line) for line in vecfile.readlines())
                val = struct.unpack('<iihh', content[:12])
                num_images = val[0]
                image_size = val[1]
                if image_size != prev_image_size:
                    err_msg = """The image sizes in the .vec files differ. These values must be the same. \n The image size of file {0}: {1}\n
                    The image size of previous files: {0}""".format(f, image_size, prev_image_size)
                    sys.exit(err_msg)

                total_num_images += num_images
        except IOError as e:
            logging.error('An IO error occured while processing the file: {0}'.format(f))
        logging.info('Processed {} / {} files'.format(total_num_images, num_files))

    # Iterate through the .vec files, writing their data (not the header) to the output file
    # '<iihh' means 'little endian, int, int, short, short'
    header = struct.pack('<iihh', total_num_images, image_size, 0, 0)
    try:
        with open(output_vec_file, 'wb') as outputfile:
            outputfile.write(header)

            for f in files:
                with open(f, 'rb') as vecfile:
                    content = ''.join(str(line) for line in vecfile.readlines())
                    data = content[12:]
                    outputfile.write(data)
    except Exception as e:
        logging.error(e)
    import shutil
    logging.info('Merge successful, deleting')
    shutil.rmtree(vec_directory)


def create_symbol_annotation_file(image_file_name, annotation_folder, api):
    annotation_file = join(annotation_folder, basename(image_file_name) + '.json')
    if os.path.isfile(annotation_file):
        return
    gray_image = cv2.imread(image_file_name, cv2.IMREAD_GRAYSCALE)
    faulty = ['5714408ea310a43cd79dfb05', '570e54e2a3103f2d963bb2df', '5726b721a310a74ad90b77d0',
              '570b1759a310ebd5d0ba530d', '5717cb68a3102c7fbf48f4bd', '572bea80a31066a71353945e',
              '5726afe1a310ebd5d0bad97b', '573a6cb4a3102f2d5d09f8cc', '57165843a31023407b767388',
              '573a3265a3106ec45f15c059', '572ce0eda310a68cf380f93f', '57082e0ea3102295b4d5b9ea',
              '5713c683a3100724fc6ab01f', '572fc5dda3102c7fbf49647e', '570ab33ea310e9ec32f94164',
              '571108e7a3100724fc6aa677', '570ab33ea310e9ec32f94164', '573cace5a310c76c3067d532',
              '570ab299a3107b84a89a53bb', '57197c47a3100724fc6ad2e8', '5739e74fa310ce1ed1a66b77',
              '571bf4afa310f2452af5ff1e', '573281fea31010683247ca73']
    if gray_image is None or get_id(image_file_name) in faulty:
        logging.error('Image {} could not be read'.format(image_file_name))
        return
    thr = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 4)
    image = PIL.Image.fromarray(thr)
    api.SetImage(image)
    boxes = api.GetComponentImages(RIL.SYMBOL, True)
    boxes = [x[1] for x in boxes]
    with open(annotation_file, 'wb') as af:
        json.dump(boxes, af)


def create_symbol_corpus(image_folder, out_folder):
    with PyTessBaseAPI() as api:
        for f in os.listdir(image_folder):
            create_symbol_annotation_file(join(image_folder, f), out_folder, api)
            logging.info('Wrote tesseract annotations for {}'.format(f))


def get_id(path):
    return basename(path).split('_')[0]

def get_symbol_boxes(image_file_name, annotation_folder):
    annotation_file = join(annotation_folder, basename(image_file_name) + '.json')
    if not os.path.exists(annotation_file):
        return None
    with open(annotation_file, 'rb') as af:
        data = json.load(af)
        return [(d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h']) for d in data]


def get_price_boxes(image_file_name, annotation_folder):
    price_re = re.compile('.*\d*\.\d+$') # really basic stuff
    annotation_file = join(annotation_folder, basename(image_file_name) + '.json')
    if not os.path.isfile(annotation_file):
        logging.warning('File {} does not exist, returning empty list'.format(annotation_file))
        return []
    with open(annotation_file) as af:
        data = json.load(af)['bboxes'][1:]
        data = [(b['top_left']['x'], b['top_left']['y'], b['bottom_right']['x'], b['bottom_right']['y'])
                for b in data if price_re.match(b['description'])]
    return data


def grab_potential_price_crops(annotation_folder, image_folder, out_folder):
    price_re = re.compile('.*\d*\.\d+$') # really basic stuff
    annotations = os.listdir(annotation_folder)
    for f in annotations:
        with open(join(annotation_folder, f)) as af:
            image_file_name = join(image_folder, basename(f)[:-5]) # just drop the .json
            base_name = basename(image_file_name)
            data = json.load(af)
            bboxes = data['bboxes'][1:] # drop the first one
            for i, b in enumerate(bboxes):
                if price_re.match(b['description']) is not None and os.path.isfile(image_file_name):
                    image_file = cv2.imread(image_file_name, cv2.IMREAD_GRAYSCALE)
                    x1, y1 = b['top_left']['x'], b['top_left']['y']
                    x2, y2 = b['bottom_right']['x'], b['bottom_right']['y']
                    # todo -> make sure you end up with a good format for training (making the vec file)
                    crop = image_file[y1: y2, x1: x2]
                    name = join(out_folder, '{}_{}'.format(i, base_name))
                    logging.info('writing {}'.format(name))
                    cv2.imwrite(name, crop)


def create_training_vector(in_file, neg_file, out_folder):
    vec_name = join(out_folder, basename(in_file)) + '.vec'
    subprocess.call(['opencv_createsamples', '-img', in_file, '-vec',  vec_name,
                     '-bg', neg_file, '-w', '72', '-h', '24', '-maxxangle', '1.4', '-maxyangle', '1.4',
                     '-maxzangle', '0.7', '-num', '500'])
    logging.info('Wrote vector for {} in {}'.format(in_file, vec_name))


def create_all_vectors(in_folder, neg_file, out_folder):
    import shutil
    if os.path.isdir(out_folder):
        logging.info('{} already present, removing and recreating'.format(out_folder))
        shutil.rmtree(out_folder)
        os.makedirs(out_folder)
    files = [join(in_folder, x) for x in os.listdir(in_folder)]
    pool = Pool(cpu_count())
    pool.map(partial(create_training_vector, neg_file=neg_file, out_folder=out_folder), files)


def tag_receipt_files(in_folder, out_folder):
    files = [os.path.join(in_folder, x) for x in os.listdir(in_folder) if 'crop' not in x and x.endswith('jpg')]
    model = cv2.CascadeClassifier('/home/gabi/workspace/opencv2_python/receipt_classifier/receipt_model.xml')
    logging.info('Loaded model')
    for f in files:
        data = cv2.imread(f)
        boxes = model.detectMultiScale(data, scaleFactor=1.05, minSize=(5, 15))
        logging.info('Found {} boxes for {}'.format(boxes, f))
        if boxes is None or len(boxes) == 0:
            logging.warning('file {} empty :('.format(f))
        else:
            for x, y, w, h in boxes:
                cv2.rectangle(data, (x, y), (x+w, y+h), (0,0,0))
        logging.info('Done tagging {}'.format(f))
        cv2.imwrite(join(out_folder, basename(f)), data)


def evaluate_untagged(in_folder, model_file, out_folder, truth_folder):
    files = [f for f in os.listdir(in_folder) if f.endswith('jpg')]
    pool = Pool(cpu_count())
    pool.map(partial(evaluate_file_with_model, in_folder=in_folder, model_file=model_file, truth_folder=truth_folder, out_folder=out_folder), files)


def evaluate_file_with_model(f, in_folder, out_folder, truth_folder, model_file):
    model = cv2.CascadeClassifier(model_file)
    image = cv2.imread(join(in_folder, f), cv2.IMREAD_GRAYSCALE)
    annot_file = join(truth_folder, f.split('.')[0] + '.json')
    if not os.path.isfile(annot_file):
        logging.info('Annotation file {} not found'.format(annot_file))
    truth_boxes = [tuple(map(int, b['bounds'])) for b in json.load(open(annot_file)) if b['entity_type'] == 'TOTAL']
    boxes = model.detectMultiScale(image, scaleFactor=1.01, minSize=(3, 9), minNeighbors=3)
    if boxes is None or len(boxes) == 0:
        logging.warning('Unable to match any boxes on {}'.format(f))
    logging.info('Found {} boxes'.format(len(boxes)))
    for tb in truth_boxes:
        closest_boxes = sorted(boxes, key=lambda b: box_distance(b, tb))[:2]
        for x, y, w, h in closest_boxes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,0), 4)
        cv2.rectangle(image, (tb[0], tb[1]), (tb[2], tb[3]), (255,0,0), 4)
    logging.info('Writing {}'.format(join(out_folder, f)))
    cv2.imwrite(join(out_folder, f), image)


def get_tesseract_symbol_crops(in_path, out_folder):
    gray_image = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        logging.error('Image {} could not be read'.format(in_path))
        return
    thr = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 4)
    image = PIL.Image.fromarray(thr)
    with PyTessBaseAPI() as api:
        api.SetImage(image)
        boxes = api.GetComponentImages(RIL.SYMBOL, True)
    boxes = [x[1] for x in boxes]
    for i, b in enumerate(boxes):
        cv2.imwrite(join(out_folder, '{}_{}'.format(i, basename(in_path))), gray_image[b['y']-1:b['y']+b['h']+1,b['x']-1:b['x']+b['w']+1])
    logging.info('wrote patches for {}'.format(in_path))


def create_positive_tesseract_corpus(in_folder, out_folder):
    pool = Pool(cpu_count())
    in_files = [join(in_folder, x) for x in os.listdir(in_folder) if x.endswith('jpg')]
    pool.map(partial(get_tesseract_symbol_crops, out_folder=out_folder), in_files)

def box_distance(b1, b2):
    return np.sqrt(sum([abs(c1 - c2) ** 2 for c1, c2 in zip(b1, b2)]))


def non_max_supression_slow(boxes, overlap_thr):
    boxes = np.array(boxes)
    if len(boxes) == 0:
        return []
    pick = []
    x1, y1, x2, y2 = tuple(boxes.T)
    area = (x2-x1+1) * (y2-y1+1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        for pos in range(last):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            xx2 = min(x2[i], x2[j])
            yy1 = max(y1[i], y1[j])
            yy2 = min(y2[i], y2[j])
            w = max(0, xx2-xx1)
            h = max(0, yy2-yy1)
            overlap = float(w * h) / area[j]
            if overlap > overlap_thr:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    return boxes[pick]


def split_image(image, crop_w, crop_h):
    h_splits = int(image.shape[0]/crop_h)
    w_splits = int(image.shape[1]/crop_w)
    crops = []
    for i in range(h_splits):
        for j in range(w_splits):
            crops.append(image[i*crop_h:(i+1)*crop_h, j*crop_w:(j+1)*crop_w])
    return crops


def get_image_crops(in_path, out_folder, crop_w, crop_h):
    i = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    crops = fast_sliding_window(i, (crop_h, crop_w), (3,3))
    for i, c in enumerate(crops):
        name = join(out_folder, '{}_{}'.format(i, basename(in_path)))
        cv2.imwrite(name, c)
    logging.info('Wrote all crops for {}'.format(in_path))


def create_vector_file(output_vector_file, neg_file, in_folder, vector_directory):
    create_all_vectors(in_folder=in_folder,
                       neg_file=neg_file, out_folder=vector_directory)
    merge_vec_files(vec_directory=vector_directory,
                    output_vec_file=output_vector_file)


def write_all_crops(in_folder, out_folder, crop_w, crop_h, file_limit):
    pool = Pool(cpu_count())
    files = [join(in_folder, i) for i in os.listdir(in_folder) if i.endswith('jpg')]
    np.random.shuffle(files)
    pool.map(partial(get_image_crops, out_folder=out_folder, crop_w=crop_w, crop_h=crop_h), files if not file_limit else files[:file_limit])


def train_model(train_data, response_data, model):
    return model.partial_fit(train_data, response_data)


def load_file_and_resize(in_path, w, h):
    data = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if data is None:
        raise Exception('Error reading {}'.format(in_path))
    if data.shape[:2] != (h, w):
        data = cv2.resize(data, (w, h), interpolation=cv2.INTER_CUBIC)
    return data


def load_file_with_skews_and_resize(in_path, w, h, skews):
    data = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if data is None:
        raise Exception('Error reading {}'.format(in_path))
    all_data = [rotate_about_center(data, s) for s in skews]
    ret_val = []
    for d in all_data:
        if d.shape != (h, w):
            ret_val.append(cv2.resize(d, (w, h)))
        else:
            ret_val.append(d)
    return ret_val


def load_training_data(positive_folder, negative_folder, width, height, save_path='train.pck'):
    if os.path.isfile(save_path):
        logging.info('Training data already present on disk')
        return pickle.load(open(save_path, 'rb'))
    logging.info('Preparing training data')
    positive_files = [join(positive_folder, x) for x in os.listdir(positive_folder) if x.endswith('jpg')]
    negative_files = [join(negative_folder, x) for x in os.listdir(negative_folder) if x.endswith('jpg')]
    pool = Pool(cpu_count())
    pos_data = list(zip(pool.map(partial(load_file_and_resize, w=width, h=height), positive_files), np.ones((len(positive_files), 1))))
    neg_data = list(zip(pool.map(partial(load_file_and_resize, w=width, h=height), negative_files), np.zeros((len(negative_files), 1))))
    all_data = [x for x in pos_data + neg_data if x[0] is not None]
    np.random.shuffle(all_data)
    all_x = [x[0] for x in all_data]
    all_y = [x[1] for x in all_data]
    pickle.dump((all_x, all_y), open(save_path, 'wb'))
    logging.info('Training data loaded and saved on disk')
    return all_x, all_y


def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(np.ceil(nw)), int(np.ceil(nh))), flags=cv2.INTER_LANCZOS4)


def batch_training_data(positive_folder, negative_folder, width, height, n_batches, skew_range=range(-60, 60, 10)):
    positive_files = [join(positive_folder, x) for x in os.listdir(positive_folder) if x.endswith('jpg')]
    negative_files = [join(negative_folder, x) for x in os.listdir(negative_folder) if x.endswith('jpg')]
    np.random.shuffle(positive_files)
    np.random.shuffle(negative_files)
    pos_batch_size = int(np.ceil(len(positive_files)/n_batches))
    neg_batch_size = int(np.ceil(len(negative_files)/n_batches))

    for i in range(n_batches):
        pos_x = positive_files[pos_batch_size*i:pos_batch_size*(i+1)]
        neg_x = negative_files[neg_batch_size*i:neg_batch_size*(i+1)]

        pos_x = [x for y in map(partial(load_file_with_skews_and_resize, w=width, h=height, skews=skew_range), pos_x) for x in y]
        pos_y = [1 for _ in pos_x]

        neg_x = list(map(partial(load_file_and_resize, w=width, h=height), neg_x))
        neg_y = [0 for _ in neg_x]
        all_x = pos_x + neg_x
        all_y = pos_y + neg_y
        yield all_x, all_y


def get_class_weight(positive_folder, negative_folder, skew_count):
    y = [1 for _ in os.listdir(positive_folder) * skew_count] + [0 for _ in os.listdir(negative_folder)]
    weights = compute_class_weight('balanced', np.array([1, 0]), y)
    return {1: weights[0], 0: weights[1]}


def prepare_model(positive_folder, negative_folder, w, h, svm_save_path='svm.pck', skews=list(range(-90, 90, 5))):
    if os.path.isfile(svm_save_path):
        return pickle.load(open(svm_save_path, 'rb'))
    model = create_model()
    first_pass = True
    model.class_weight = get_class_weight(positive_folder, negative_folder, skew_count=len(skews))
    for i, (all_x, all_y) in enumerate(batch_training_data(positive_folder=positive_folder,
                                                           negative_folder=negative_folder,
                                                           width=w,
                                                           height=h,
                                                           n_batches=50,
                                                           skew_range=skews)):
        all_x = np.array([hog(x).flatten() for x in all_x]).astype(np.float32)
        all_y = np.array(all_y).astype(np.int32).ravel()
        logging.info('Training model on {} samples at batch {}'.format(len(all_y), i+1))
        if first_pass:
            model = model.partial_fit(all_x, all_y, classes=[1, 0])
            first_pass = False
        else:
            model = model.partial_fit(all_x, all_y)
    pickle.dump(model, open(svm_save_path, 'wb'))
    logging.info('Model saved on disk')
    return model


def create_model():
    model = SGDClassifier()
    return model


def fast_sliding_window(image, win_size, win_stride):
    slides = extract_patches(image, patch_shape=win_size, extraction_step=win_stride)
    for i in range(slides.shape[0]):
        for j in range(slides.shape[1]):
            yield slides[i, j]


def sliding_window(image, win_size, win_stride):
    image_h, image_w = image.shape
    slide_h, slide_w = win_size
    stride_h, stride_w = win_stride
    if image_h < slide_h and image_w < slide_w:
        yield (0, 0), image.shape
    else:
        h_limit = int(ceil((image_h-slide_h)/stride_h))
        w_limit = int(ceil((image_w-slide_w)/stride_w))
        for i in range(h_limit):
            for j in range(w_limit):
                yield (i*stride_h, j*stride_w) ,(i*stride_h+slide_h, j*stride_w+slide_w)


def image_pyramid_up(image, scale_factor, max_w, max_h):
    # we do not need to return the original
    while True:
        new_w = int(image.shape[1] * scale_factor)
        new_h = int(image.shape[0] * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        yield image
        if new_h > max_h or new_w > max_w:
            break


def image_pyramid_down(image, scale_factor, min_w, min_h):
    # we do not need to return the original
    while True:
        new_w = int(image.shape[1] / scale_factor)
        new_h = int(image.shape[0] / scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        yield image
        if new_h < min_h or new_w < min_w:
            break


def detect_boxes(image, svm, win_size, win_stride):
    positives = []
    for (y1, x1), (y2, x2) in sliding_window(image, win_size, win_stride):
        crop = image[y1:y2, x1:x2]
        descriptor = hog(crop)
        result = svm.predict(descriptor.reshape(1, len(descriptor)))
        if result[0] == 1.:
            positives.append(((y1, x1), (y2, x2)))
    return positives


def detect_multi_scale(image, svm, scale, win_size, win_stride, max_size=None, min_size=None):
    # TODO -> fix max and min sizes to a some more intuitive value
    if max_size is None:
        max_h, max_w = int(image.shape[0] * 2), int(image.shape[1] * 2)
    else:
        max_h, max_w = max_size
    if min_size is None:
        min_h, min_w = int(image.shape[0]/2), int(image.shape[1]/2)
    else:
        min_h, min_w = min_size

    # first the normal image

    # TODO if we ditch the generator pattern and just man up and return a list from up/down scaling, we can then
    # TODO use multiprocessing to evaluate these in parallel.. maybe even merge the scale/evaluate steps into one
    down = list(image_pyramid_down(image, scale, min_h, min_w))
    up = list(image_pyramid_up(image, scale, max_h, max_w))
    logging.debug('Detecting on {} scaled images'.format(len(up) + len(down) + 1))

    results = detect_boxes(image, svm, win_size, win_stride)
    # then down scaled
    for i in down:
        down_results = detect_boxes(i, svm, win_size, win_stride)
        if len(down_results) > 0:
            up_scale = image.shape[0] / i.shape[0]
            down_results = [((y1*up_scale, x1*up_scale), (y2*up_scale, x2*up_scale)) for ((y1, x1), (y2, x2)) in down_results]
            results.extend(down_results)

    # then up scaled
    for i in up:
        up_results = detect_boxes(i, svm, win_size, win_stride)
        if len(up_results) > 0:
            down_scale = image.shape[0] / i.shape[0]
            up_results = [((y1*down_scale, x1*down_scale), (y2*down_scale, x2*down_scale)) for
                          ((y1, x1), (y2, x2)) in up_results]
            results.extend(up_results)

    return results


def evaluate_and_return_results(in_path,
                                model,
                                win_size,
                                win_stride,
                                scale,
                                truth_folder):
    then = time.time()
    annot_file = join(truth_folder, basename(in_path).split('.')[0] + '.json')
    truth_boxes = [tuple(map(int, b['bounds'])) for b in json.load(open(annot_file)) if b['entity_type'] == 'TOTAL']
    image = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    results = detect_multi_scale(image=image,
                                 svm=model,
                                 scale=scale,
                                 win_size=win_size,
                                 win_stride=win_stride)
    # apply non maxima suppression
    results = [(x1, y1, x2, y2) for (y1, x1), (y2, x2) in results]
    results = [tuple(x) for x in non_max_supression_slow(results, 0.8)]
    tp = 0

    matched = set()
    for t in truth_boxes:
        true = [x for x in results if intersection_over_union(t, x) >= 0.5]
        if len(true) > 0:
            # increase tp
            tp += 1
        for tr in true:
            matched.add(tr)
    return tp, len(truth_boxes), len(results), time.time() - then


def evaluate_and_copy_negatives(in_path,
                                svm,
                                win_size,
                                win_stride,
                                scale,
                                truth_folder,
                                negative_folder):
    truth_boxes = get_price_boxes(in_path, truth_folder)
    if truth_boxes is None:
        logging.error('No annotation file present for {}...skipping analisys'.format(in_path))
    image = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    then = time.time()
    results = detect_multi_scale(image=image,
                                 svm=svm,
                                 scale=scale,
                                 win_size=win_size,
                                 win_stride=win_stride)
    elapsed = time.time() - then
    print('Detection took {} seconds'.format(elapsed))
    if len(results) > 0:
        logging.info('Found {} results for file {}'.format(len(results), in_path))
        new_boxes = [(x1, y1, x2, y2) for ((y1, x1), (y2, x2)) in results]
        true_results = set()
        for tb in truth_boxes:
            t = [b for b in new_boxes if intersection_over_union(b, tb) > 0.2]
            for tr in t:
                true_results.add(tr)
        false_results = [x for x in new_boxes if x not in true_results and box_area(*x) > 0]
        logging.info('out of {} boxes, {} are classified as negatives and {} as positives'.
                     format(len(new_boxes), len(false_results), len(true_results)))
        for i, b in enumerate(false_results):
            x1, y1, x2, y2 = tuple(b)
            crop = image[y1:y2, x1:x2]
            try:
                crop = cv2.resize(crop, (win_size[1], win_size[0]), interpolation=cv2.INTER_CUBIC)
            except:
                logging.error('Error resizing from {}'.format(crop.shape))
            fp_name = join(negative_folder, '{}_{}'.format(i, basename(in_path)))
            cv2.imwrite(join(fp_name), crop)
        logging.info('Wrote {} false positives to {}'.format(len(false_results), negative_folder))


def evaluate_and_write_results(in_path,
                               svm,
                               win_size,
                               win_stride,
                               scale,
                               out_folder):
    image = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    results = detect_multi_scale(image=image,
                                 svm=svm,
                                 scale=scale,
                                 win_size=win_size,
                                 win_stride=win_stride)
    if len(results) == 0:
        print('No results found for {}'.format(in_path))
        return
    logging.info('Found {} results for file {}'.format(len(results), in_path))
    new_boxes = [(x1, y1, x2, y2) for ((y1, x1), (y2, x2)) in results]
    for x1, y1, x2, y2 in new_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), 0, 2)
    cv2.imwrite(join(out_folder, basename(in_path)), image)
    logging.info('Done evaluating {}'.format(in_path))


def box_area(x1, y1, x2, y2):
    return abs(x2-x1)*abs(y2-y1)


def write_corners(in_path, out_folder):
    orig = cv2.imread(in_path)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype('float32')
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    # dst = cv2.dilate(dst, None, 3)
    orig[dst>0.005*dst.max()]=[0,0,255]
    cv2.imwrite(join(out_folder, basename(in_path)), orig)
    logging.info('Wrote corners for {}'.format(in_path))


def intersection_over_union(b1, b2):
    """
        Calculate intersection / union ratio between 2 rectangles
        Returns a positive value (between 0 and 1) representing
        the ratio between the 2 areas or 0 if the boxes have no intersecting area
    """
    # if they don't intersect
    if b1[2] < b2[0] or b1[3] < b2[1]:
        return 0
    union_p1 = (min(b1[0], b2[0]), min(b1[1], b2[1]))
    union_p2 = (max(b1[2], b2[2]), max(b1[3], b2[3]))

    intersect_p1 = (max(b1[0], b2[0]), max(b1[1], b2[1]))
    intersect_p2 = (min(b1[2], b2[2]), min(b1[3], b2[3]))

    union_area = (union_p2[0] - union_p1[0]) * (union_p2[1] - union_p1[1])
    intersect_area = (intersect_p2[0] - intersect_p1[0]) * (intersect_p2[1] - intersect_p1[1])
    if union_area <= 0 or intersect_area <= 0:
        return 0
    return float(intersect_area) / float(union_area)


def evaluate_folder(positive_folder,
                    negative_folder,
                    w,
                    h,
                    image_folder,
                    scale=1.05,
                    stride=(3, 3)):
    model = prepare_model(positive_folder=positive_folder, negative_folder=negative_folder, w=w, h=h)
    pool = Pool(cpu_count())
    files = [join(image_folder, x) for x in os.listdir(image_folder) if x.endswith('jpg')]
    results = pool.map(partial(evaluate_and_return_results,
                               model=model,
                               win_size=(h,w),
                               win_stride=stride,
                               scale=scale,
                               truth_folder='/home/gabi/workspace/eloquentix/benchmark/annotations'), files)
    tp, ap, ar, at = tuple(zip(*results))
    tp, ap = sum(tp), sum(ap)
    fn = ap - tp
    recall = tp / (tp + fn)
    avg_time = np.mean(at)
    print('Recall score: {}, Avg time: {}, Max time: {}, Min time: {}'.format(recall, avg_time, max(at), min(at)))
    pickle.dump(results, open('/tmp/results.pck'))

def eval_one():
    model = prepare_model(None, None, 0, 0)
    r = evaluate_and_return_results(in_path='/home/gabi/workspace/eloquentix/benchmark/images/57195cfea3103e0c1852259b_848dd4e9-fbb1-482c-aea8-e7dd2f4d53f6.jpg',
                                model=model,
                                win_size=(20, 80),
                                win_stride=(3, 3),
                                scale=1.05,
                                truth_folder='/home/gabi/workspace/eloquentix/benchmark/annotations')
    print(r)

def write_all_file_corners():
    image_folder = '/home/gabi/workspace/eloquentix/image-corpus/images'
    out_folder = '/tmp/out'
    for f in os.listdir(image_folder):
        write_corners(join(image_folder, f), out_folder)


def evaluate_full():
    # positive_folder = '/home/gabi/workspace/opencv-haar-classifier-training/positive_crops/'
    # negative_folder = '/home/gabi/workspace/opencv-haar-classifier-training/negative_crops'
    image_folder = '/home/gabi/workspace/eloquentix/benchmark/images'
    w = 80
    h = 20
    evaluate_folder(positive_folder=None,
                    negative_folder=None,
                    w=w,
                    h=h,
                    image_folder=image_folder)


def mask_image(image, mask_size):
    mask = np.zeros_like(image)
    for i in range(int(mask.shape[0]/mask_size)):
        for j in range(int(mask.shape[1]/mask_size)):
            val = int(np.mean(image[i*mask_size:(i+1)*mask_size, j*mask_size:(j+1)*mask_size]))
            mask[i*mask_size:(i+1)*mask_size, j*mask_size:(j+1)*mask_size] = val
    return mask


def image_center_and_total_mean_and_median(in_path):
    radius = 25
    i = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if i is None:
        return None
    h, w = i.shape
    h_center, w_center = int(h/2), int(w/2)
    y1, x1 = 0 if h_center<radius else h_center-radius, 0 if w_center<radius else w_center-radius
    y2, x2 = min(h, h_center+radius), min(w, w_center + radius)
    crop = i[y1:y2,x1:x2]
    image_mean, image_median = np.mean(i), np.median(i)
    crop_mean, crop_median = np.mean(crop), np.median(crop)
    return image_mean, image_median, crop_mean, crop_median


if __name__ == '__main__':
    # write_all_file_corners()
    # create_symbol_corpus('/home/gabi/workspace/eloquentix/image-corpus/images',
    #                      '/home/gabi/workspace/eloquentix/image-corpus/tesseract')
    evaluate_full()
    # evaluate_full()
    # write_all_crops(out_folder='/home/gabi/workspace/opencv-haar-classifier-training/negative_symbols',
    #                 in_folder='/home/gabi/workspace/opencv-haar-classifier-training/negative_images/',
    #                 crop_h=16, crop_w=8, file_limit=None)
    # create_vector_file(in_folder='/home/gabi/workspace/opencv-haar-classifier-training/positive_images/',
    #                    neg_file='/home/gabi/workspace/opencv-haar-classifier-training/negatives.txt',
    #                    output_vector_file='/home/gabi/workspace/opencv-haar-classifier-training/total_vector.vec',
    #                    vector_directory='/home/gabi/workspace/opencv-haar-classifier-training/samples')
    # evaluate_untagged(in_folder='/home/gabi/workspace/eloquentix/benchmark/images',
    #                   model_file='/home/gabi/workspace/opencv2_python/cascade.xml',
    #                   out_folder='/tmp/total_tags',
    #                   truth_folder='/home/gabi/workspace/eloquentix/benchmark/annotations')