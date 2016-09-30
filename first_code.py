import cv2, numpy as np
import time

def camera_detect():
    model = cv2.CascadeClassifier('/home/gabi/opencv_contrib/modules/face/data/cascades/haarcascade_mcs_righteye.xml')
    cap = cv2.VideoCapture(0)
    window = cv2.namedWindow('frame')
    times = []
    take_it = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, -1, 1, 0)
        grad_y = cv2.Sobel(gray, -1, 0, 1)
        grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        # canny = cv2.Canny(gray, 40, 150)
        _, grad_contours, _ = cv2.findContours(grad, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # _, contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # new_contours = [cv2.approxPolyDP(c, float(cv2.arcLength(c, True))/50., True) for c in grad_contours]
        # mask = np.zeros_like(gray)
        # cv2.drawContours(mask, new_contours, -1, 255)
        # cv2.drawContours(mask, grad_contours, -1, 180)
        cv2.imshow('frame', grad_y)
        take_it += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    camera_detect()