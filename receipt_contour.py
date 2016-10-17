import cv2
import numpy as np
from multiprocessing import cpu_count, Pool
from os.path import basename, join
from os import listdir
from functools import partial
from swt import SWTScrubber
from tesserocr import RIL, PyTessBaseAPI
import PIL


class SwtPoint2D:
    def __init__(self, x, y, swt):
        self.x = x
        self.y = y
        self.swt = swt




def get_receipt_content(gray_image):
    blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
    edges = cv2.Canny(blur, 10, 100, L2gradient=True)
    _, cnt, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnt, key=lambda x: cv2.contourArea(x), reverse=True)
    screen_contour = None
    for c in cnt[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, False)
        if len(approx) < 10:
            screen_contour = approx
            break
    if screen_contour is not None:
        cv2.drawContours(edges, [screen_contour], -1, 255, 2)
    return edges


def get_receipt_crop(gray_image):
    iterations = 0
    fade = np.copy(gray_image)
    while True:
        hist, bins = np.histogram(fade, bins=3)
        if (hist[0] + hist[2]) / hist[1] > 9:
            break
        if iterations > 200:
            print(':(')
            break
        iterations += 1
        fade = cv2.dilate(fade, kernel=np.ones((5, 5)))

    thr = cv2.adaptiveThreshold(fade, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 4)
    mask = np.zeros_like(gray_image)
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if thr[i][j] == 255:
                mask[i][j] = gray_image[i][j]
    return mask


def get_text_blocks(gray_image):
    # blur = cv2.GaussianBlur(gray_image, (7, 7), 0)
    edges = cv2.Canny(gray_image, 50, 200)
    dilated = cv2.dilate(edges, kernel=np.ones((1,3)), iterations=5)
    _, contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_with_boxes = [(c, cv2.minAreaRect(c)) for c in contours]
    angle_histogram = np.histogram([x[1][-1] for x in contours_with_boxes])
    return dilated

def get_receipt_blocks(gray_image):
    massive_blur = cv2.dilate(gray_image, np.ones((7,7)), iterations=30)
    _, hist = np.histogram(massive_blur, bins=2)
    _, paper_edge = cv2.threshold(massive_blur, hist[1], 255, cv2.THRESH_BINARY)
    return paper_edge

def get_sobel_grads(gray_image):
    blur = cv2.GaussianBlur(gray_image, (7,7), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(blur, cv2.CV_16S, 0, 1)
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
    return cv2.dilate(grad,np.ones((1,3)), iterations=4)


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, 1.0 - sigma) *v)
    upper = int(min(255, 1.0 + sigma) *v)
    return cv2.Canny(image, lower, upper)


def get_channels(image):
    if len(image.shape) != 3:
        raise Exception()
    return image[:,:, 0], image[:, :, 1], image[:, :, 2]


def write_cropped_receipt(in_path, out_folder):
    image = cv2.imread(in_path)[:, :, 0]
    cv2.imwrite(join(out_folder, basename(in_path)), get_receipt_crop(image))
    print('Done writing {}'.format(in_path))


def write_sobel(in_path, out_folder):
    image = cv2.imread(in_path)[:,:,0]
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(join(out_folder, basename(in_path)), get_sobel_grads(blur))
    print('Done writing {}'.format(in_path))


def write_thr_sobel(in_path, out_folder):
    i = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(join(out_folder, basename(in_path)), threshold_deriv(i))
    print('Done writing {}'.format(in_path))


def threshold_deriv(gray_image):
    blur = cv2.GaussianBlur(gray_image, ksize=(3, 3), sigmaX=0)
    grad_x = cv2.Sobel(blur, -1, 1, 0)
    grad_y = cv2.Sobel(blur, -1, 0, 1)
    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0.)
    histogram = np.histogram(grad, bins=25)
    print(histogram)
    thr = cv2.adaptiveThreshold(grad, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 4)
    return grad


def write_hsv_image_channels(in_path, out_folder):
    image = cv2.imread(in_path)
    base = basename(in_path).split('.')[0]
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2)
    # cv2.imwrite(join(out_folder, base) + '_H.jpg', yuv[:,:,0])
    cv2.imwrite(join(out_folder, base) + '_S.jpg', yuv[:,:,1])
    # cv2.imwrite(join(out_folder, base) + '_V.jpg', yuv[:,:,2])
    print('Done writing {}'.format(in_path))


def write_yuv_image_channels(in_path, out_folder):
    image = cv2.imread(in_path)
    base = basename(in_path).split('.')[0]
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    u = yuv[:,:,1]
    u = cv2.Canny(u, 0, 20)
    cv2.imwrite(join(out_folder, base) + '_U.jpg', u)
    print('Done writing {}'.format(in_path))



def text_detect(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    # relax the edge detection a bit
    edges = cv2.Canny(image, 175, 320)
    grad_x = cv2.Scharr(blur, -1, 1, 0)
    grad_y = cv2.Scharr(blur, -1, 0, 1)
    # Blur the gradients
    grad_x = cv2.GaussianBlur(grad_x, (3, 3), 0)
    grad_y = cv2.GaussianBlur(grad_y, (3, 3), 0)


def stroke_width_transform(edges, grad_x, grad_y):
    swt_image = np.ones_like(edges) * np.inf
    rays = []
    precision = 0.05

def write_rgb_image_channels(in_path, out_folder):
    image = cv2.imread(in_path)
    r, g, b = get_channels(image)
    base = basename(in_path).split('.')[0]
    cv2.imwrite(join(out_folder, base) + '_R.jpg', r)
    cv2.imwrite(join(out_folder, base) + '_G.jpg', g)
    cv2.imwrite(join(out_folder, base) + '_B.jpg', b)
    print('Done writing {}'.format(in_path))


def contour_stats(in_path, out_folder):
    img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    canny = cv2.Canny(blur, 10, 30)
    _, contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.minAreaRect(c) for c in contours]


    filtered_contours_and_boxes = [(c, boxes[i]) for i, c in enumerate(contours) if filter_box(boxes[i], img.shape, c)]

    filtered_contours = [f[0] for f in filtered_contours_and_boxes]
    filtered_boxes = [f[1] for f in filtered_contours_and_boxes]
    counts, angles = np.histogram([x[2] for x in filtered_boxes], bins=6)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, filtered_contours, -1, 255, 1)
    cv2.imwrite(join(out_folder, basename(in_path)), mask)


def filter_box(bbox, orig_shape, contour):
    b_w, b_h =  bbox[1]
    i_w, i_h = orig_shape
    c_area = cv2.contourArea(contour)
    c_peri = cv2.arcLength(contour, True)
    if c_area == 0 or c_peri == 0:
        return False
    box_peri = 2 * b_w + 2 * b_h
    box_area = b_w * b_h
    # if box_area / c_area > 10:
    #     return False
    # if max(box_peri, c_peri) / min(box_peri, c_peri) > 10:
    #     return False
    if b_w == 0 or b_h == 0:
        return False
    if not(4 < b_w < 100) or not(4 < b_h < 100):
        return False
    if max(*bbox[1]) / min(*bbox[1]) > 40:
        return False
    if max(*bbox[1]) > min(*orig_shape) / 2:
        return False
    if max(b_w, b_h) < 8:
        return False
    return True

def get_contours_of_interest(gray_image):
    edges = cv2.Canny(gray_image, 30, 100)
    # while True:
    # edges = cv2.dilate(edges, kernel=np.ones((1,3)), iterations=1)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    hulls = []
    for c in contours:
        hull = cv2.convexHull(c)
        (x, y), (h, w), angle = cv2.minAreaRect(hull)
        if w < 4 or h < 4:
            continue
        if min(h, w) > max(*gray_image.shape) / 2:
            continue
        if max(w, h) / min(w, h) > 100:
            continue

        hulls.append(hull)
    mask = np.zeros_like(gray_image)
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            for hull in hulls:
                if cv2.pointPolygonTest(hull, (c, r), False) >= 0:
                    mask[r][c] = gray_image[r][c]
                    continue
    return mask


def write_contours(in_path, out_folder):
    i = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(join(out_folder, basename(in_path)), get_contours_of_interest(i))
    print('Done writing {}'.format(in_path))


def write_swt_image(in_path, out_folder):
    orig = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    try:
        mask = SWTScrubber.scrub(in_path)
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if mask[r][c] != 0:
                    mask[r][c] = orig[r][c]
    # out_mask = mask * 5
        cv2.imwrite(join(out_folder, basename(in_path)), mask)
    except:
        import traceback
        traceback.print_exc()
        print(in_path)

def write_all_from_folder(in_folder, out_folder):
    pool = Pool(cpu_count())
    in_files = [join(in_folder, x) for x in listdir(in_folder) if x.endswith('jpg') and x.startswith('570a')]
    pool.map(partial(contour_stats, out_folder=out_folder), in_files)
    # for f in in_files:
    #     write_swt_image(f, out_folder)

if __name__ == '__main__':
    pass