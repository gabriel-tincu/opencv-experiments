import json
import os
import struct
import sys
import traceback
import glob
import cv2
from os.path import basename, join
import logging
import re

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


def exception_response(e):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    for line in lines:
        print(line)


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
        print('An IO error occured while processing the file: {0}'.format(files[0]))
        exception_response(e)

    # Get the total number of images
    total_num_images = 0
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
            exception_response(e)

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


def grab_potential_price_crops(annotation_folder, image_folder, out_folder):
    price_re = re.compile('\d*\.\d+') # relly basic stuff
    annotations = os.listdir(annotation_folder)
    for f in annotations:
        with open(f) as af:
            image_file_name = join(image_folder, basename(f)[:-5]) # just drop the .json
            data = json.load(af)
            bboxes = data['bboxes'][1:] # drop the first one
            for i, b in enumerate(bboxes):
                if price_re.search(b['description']) is not None:
                    image_file = cv2.imread(image_file_name, cv2.IMREAD_GRAYSCALE)
                    x1, y1 = tuple(b['top_left'])
                    x2, y2 = tuple(b['bottom_right'])
                    # todo -> make sure you end up with a good format for training (making the vec file)
                    crop = image_file[y1: y2, x1: x2]
                    cv2.imwrite(join(out_folder, '{}_{}'.format(i, image_file_name)), crop)


def tag_receipt_files(in_folder, out_folder):
    files = [os.path.join(in_folder, x) for x in os.listdir(in_folder) if 'crop' not in x and x.endswith('jpg')]
    model = cv2.CascadeClassifier('/home/gabi/workspace/opencv2_python/receipt_classifier/receipt_model.xml')
    logging.info('Loaded model')
    for f in files:
        data = cv2.imread(f)
        boxes = model.detectMultiScale(data, scaleFactor=1.3, minSize=(50, 50))
        logging.info('Found {} boxes for {}'.format(boxes, f))
        if boxes is None or len(boxes) == 0:
            logging.warning('file {} empty :('.format(f))
        else:
            for x, y, w, h in boxes:
                cv2.rectangle(data, (x, y), (x+w, y+h), (0,0,0))
        logging.info('Done tagging {}'.format(f))
        cv2.imwrite(join(out_folder, basename(f)), data)

if __name__ == '__main__':
    tag_receipt_files('/home/gabi/workspace/eloquentix/image-corpus/images', '/tmp/nou_annotated')
