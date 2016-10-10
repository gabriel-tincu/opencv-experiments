import pickle
import json
import os
import struct
import sys
import traceback
import glob
import cv2
from os.path import basename, join
from multiprocessing import Pool, cpu_count
import logging
import re
import subprocess
import numpy as np
from functools import partial
from sklearn.svm import LinearSVC
from skimage.feature import hog
from math import ceil
logging.basicConfig(level=logging.DEBUG)

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
        exception_response(e)
    import shutil
    logging.info('Merge successful, deleting')
    shutil.rmtree(vec_directory)


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
    truth_boxes = [list(map(int, b['bounds'])) for b in json.load(open(annot_file)) if b['entity_type'] == 'TOTAL']
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
    i = cv2.imread(in_path)
    crops = split_image(i, crop_w, crop_h)
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
    pool.map(partial(get_image_crops, out_folder=out_folder, crop_w=crop_w, crop_h=crop_h), files[:file_limit])


def train_svm(train_data, response_data, svm):
    svm.fit(train_data, response_data)
    return svm


def load_file_and_resize(in_path, w, h):
    data = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if data is None:
        raise Exception('Error reading {}'.format(in_path))
    if data.shape[:2] != (h, w):
        data = cv2.resize(data, (w, h))
    return data


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
    logging.info('Training data loaded and savedlogging.info on disk')
    return all_x, all_y


def prepare_svm(positive_folder, negative_folder, w, h, svm_save_path='svm.pck'):
    if os.path.isfile(svm_save_path):
        return pickle.load(open(svm_save_path, 'rb'))
    svm = create_svm()
    all_x, all_y = load_training_data(positive_folder=positive_folder,
                                      negative_folder=negative_folder,
                                      width=w,
                                      height=h)
    all_x = np.array([hog(x, pixels_per_cell=(h/4, h/4), cells_per_block=(4, 4)).flatten() for x in all_x]).astype(np.float32)
    all_y = np.array(all_y).astype(np.int32).ravel()
    logging.info('Training SVM on {} samples'.format(len(all_y)))
    if not os.path.isfile('svm.pck'):
        svm = train_svm(all_x, all_y, svm)
    pickle.dump(svm, open(svm_save_path, 'wb'))
    logging.info('SVM saved on disk')
    return svm


def create_svm():
    svm = LinearSVC(C=3)
    return svm


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
        image = cv2.resize(image, (new_h, new_w))
        yield image
        if new_h > max_h or new_w > max_w:
            break


def image_pyramid_down(image, scale_factor, min_w, min_h):
    # we do not need to return the original
    while True:
        new_w = int(image.shape[1] / scale_factor)
        new_h = int(image.shape[0] / scale_factor)
        image = cv2.resize(image, (new_h, new_w))
        yield image
        if new_h < min_h or new_w < min_w:
            break


def detect_boxes(image, svm, win_size, win_stride):
    positives = []
    win_h, win_w = win_size
    for (y1, x1), (y2, x2) in sliding_window(image, win_size, win_stride):
        crop = image[y1:y2, x1:x2]
        descriptor = hog(crop, pixels_per_cell=(win_h/4, win_h/4), cells_per_block=(4, 4)).flatten()
        result = svm.predict(descriptor.reshape(1, len(descriptor)))
        if result[0] == 1.:
            positives.append(((y1, x1), (y2, x2)))
    return positives


def detect_multi_scale(image, svm, scale, win_size, win_stride, max_size=None, min_size=None):
    if max_size is None:
        max_h, max_w = image.shape[0] * 2, image.shape[1] * 2
    else:
        max_h, max_w = max_size
    if min_size is None:
        min_h, min_w = int(image.shape[0]/2), int(image.shape[1]/2)
    else:
        min_h, min_w = min_size

    # first the normal image
    results = detect_boxes(image, svm, win_size, win_stride)

    # TODO if we ditch the generator pattern and just man up and return a list from up/down scaling, we can then
    # TODO use multiprocessing to evaluate these in parallel.. maybe even merge the scale/evaluate steps into one

    # then down scaled
    for i in image_pyramid_down(image, scale, min_h, min_w):
        down_results = detect_boxes(i, svm, win_size, win_stride)
        if len(down_results) > 0:
            up_scale = image.shape[0] / i.shape[0]
            down_results = [((y1*up_scale, x1*up_scale), (y2*up_scale, x2*up_scale)) for ((y1, x1), (y2, x2)) in down_results]
            results.extend(down_results)

    # then up scaled
    for i in image_pyramid_up(image, scale, max_h, max_w):
        up_results = detect_boxes(i, svm, win_size, win_stride)
        if len(up_results) > 0:
            down_scale = image.shape[0] / i.shape[0]
            up_results = [((y1*down_scale, x1*down_scale), (y2*down_scale, x2*down_scale)) for ((y1, x1), (y2, x2)) in up_results]
            results.extend(up_results)

    return results


def evaluate_and_copy_negatives(in_path, svm, win_size, win_stride, scale, truth_folder, negative_folder):
    truth_boxes = get_price_boxes(in_path, truth_folder)
    image = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    results = detect_multi_scale(image=image,
                                 svm=svm,
                                 scale=scale,
                                 win_size=win_size,
                                 win_stride=win_stride)
    if len(results) > 0:
        logging.info('Found {} results for file {}'.format(len(results), in_path))
        new_boxes = [(x1, y1, x2, y2) for ((y1, x1), (y2, x2)) in results]
        true_results = set()
        for tb in truth_boxes:
            t = [b for b in new_boxes if intersection_over_union(b, tb) > 0.5]
            for tr in t:
                true_results.add(tr)
        false_results = [x for x in new_boxes if x not in true_results]
        logging.info('out of {} boxes, {} are classified as negatives and {} as positives'.format(len(new_boxes), len(false_results), len(true_results)))
        for i, b in enumerate(false_results):
            x1, y1, x2, y2 = tuple(b)
            crop = cv2.resize(image[y1:y2, x1:x2], win_size)
            fp_name = join(negative_folder, '{}_{}'.format(i, basename(in_path)))
            logging.info('Writing fp {}'.format(fp_name))
            cv2.imwrite(join(fp_name), crop)


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
                    truth_folder,
                    scale=1.05,
                    stride=(3, 3)):
    svm = prepare_svm(positive_folder=positive_folder, negative_folder=negative_folder, w=w, h=h)
    pool = Pool(cpu_count())
    files = [join(image_folder, x) for x in os.listdir(image_folder) if x.endswith('jpg')]
    pool.map(partial(evaluate_and_copy_negatives,
                     svm=svm,
                     win_size=(h,w),
                     win_stride=stride,
                     scale=scale,
                     truth_folder=truth_folder,
                     negative_folder=negative_folder), files)


def evaluate_full():
    truth_folder = '/home/gabi/workspace/eloquentix/image-corpus/google'
    positive_folder = '/home/gabi/workspace/opencv-haar-classifier-training/positive_images/'
    negative_folder = '/home/gabi/workspace/opencv-haar-classifier-training/negative_crops'
    image_folder = '/home/gabi/workspace/eloquentix/image-corpus/images'
    w = 60
    h = 20
    evaluate_folder(positive_folder=positive_folder,
                    negative_folder=negative_folder,
                    w=w,
                    h=h,
                    image_folder=image_folder,
                    truth_folder=truth_folder)

if __name__ == '__main__':
    evaluate_full()

    # write_all_crops(out_folder='/home/gabi/workspace/opencv-haar-classifier-training/negative_text_crops',
    #                 in_folder='/home/gabi/workspace/opencv-haar-classifier-training/all_negatives/',
    #                 crop_h=20, crop_w=60, file_limit=1000)
    # create_vector_file(in_folder='/home/gabi/workspace/opencv-haar-classifier-training/positive_images/',
    #                    neg_file='/home/gabi/workspace/opencv-haar-classifier-training/negatives.txt',
    #                    output_vector_file='/home/gabi/workspace/opencv-haar-classifier-training/total_vector.vec',
    #                    vector_directory='/home/gabi/workspace/opencv-haar-classifier-training/samples')
    # evaluate_untagged(in_folder='/home/gabi/workspace/eloquentix/benchmark/images',
    #                   model_file='/home/gabi/workspace/opencv2_python/cascade.xml',
    #                   out_folder='/tmp/total_tags',
    #                   truth_folder='/home/gabi/workspace/eloquentix/benchmark/annotations')
