"""
1. create aligned image data in a directory
2. create a csv file that contains aligned image path, facial keypoints, original path,
"""

import dlib
import cv2
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.aligners import ResizeAligner

DATA_DIR = '/hdd1/data/PAIN/'
FACE_DATA_DIR = '/hdd1/data/PAIN/processed/face_crops/images/'
CSV_PATH = '/hdd1/data/PAIN/processed/face_crops/data.csv'
W, H = 128, 128

PROTOTXT_PATH = "/home/harry/models/opencv_face/deploy.prototxt.txt"
FACE_MODEL_PATH = "/home/harry/models/opencv_face/res10_300x300_ssd_iter_140000.caffemodel"
DLIB_68m_PATH = "/home/harry/models/dlib_face/shape_predictor_68_face_landmarks.dat"
DLIB_5m_PATH = "/home/harry/models/dlib_face/shape_predictor_5_face_landmarks.dat"
detect_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, FACE_MODEL_PATH)
detector_dlib = dlib.get_frontal_face_detector()
predictor_5 = dlib.shape_predictor(DLIB_5m_PATH)
predictor_68 = dlib.shape_predictor(DLIB_68m_PATH)

# dlib facial keypoints
# LEFT_EYE_INDICES = list(range(36, 42))
# RIGHT_EYE_INDICES = list(range(42, 48))
LEFT_EYE_INDICES = list(range(42, 48))
RIGHT_EYE_INDICES = list(range(36, 42))
# TODO: aam, aus


def crop_face(img, detector='cv2'):
    """
    :param img: numpy.array; bgr, loaded by cv2
    :return img_face: numpy.array; face part in the image, bgr
    :return (t,l,b,r): tuple; int, top, left, bottom, right
    """
    (h, w) = img.shape[:2]
    dlib_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    l, t, r, b = (0, 0, h, w)
    if detector == 'cv2':
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        detect_net.setInput(blob)
        detections = detect_net.forward()
        max_face_area = 0
        max_face_rect = None
        if detections.shape[2] != 0:
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence < 0.85:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if startX < 0 or startY < 0 or endX > w or endY > h:
                    continue
                face_area = (endX - startX) * (endY - startY)
                if face_area > max_face_area:
                    max_face_rect = [startX, startY, endX, endY]
                    max_face_area = face_area
                    l, t, r, b = max_face_rect
        # print(max_face_rect)
        # bbox = dlib.rectangle()

    elif detector == 'dlib':
        dets = detector_dlib(dlib_img, 1)
        # print("number of face detected {}...".format(len(dets)))
        bbox = dets[0]
        t, l, b, r = bbox.top(), bbox.left(), bbox.bottom(), bbox.right()
    else:
        raise ValueError("detector {} not understood! It has to be in (dlib, cv2)".format(detector))

    img_face = img[t:b, l:r]

    # if draw_landmarks:
    #     bbox = dlib.rectangle(l, t, r, b)
    #     land_marks = predictor_68(dlib_img, bbox)
    #     for i, p in enumerate(land_marks.parts()):
    #         # draw certain landmarks
    #         # if i in list(range(37-1,42)):
    #         if i in RIGHT_EYE_INDICES:
    #             x, y = p.x, p.y
    #             mat_x, mat_y = y - t, x - l
    #             img_face[mat_x - 1:mat_x + 1, mat_y - 1:mat_y + 1] = (0, 255, 0)

    return img_face, (t, l, b, r)


def get_landmarks(img, top, left, bottom, right):
    """ get facial landmarks WITHIN FACE, i.e. landmarks coords are within face img
    :param img: numpy.array; rgb image
    :param top: int;
    :param left: int;
    :param bottom: int;
    :param right: int;
    :return: list; list of int facial land marks: [x0, y0, x1, y1,..., x67, y67]
    """
    landmarks_list = []
    bbox = dlib.rectangle(left, top, right, bottom)
    land_marks = predictor_68(img, bbox)
    for i, p in enumerate(land_marks.parts()):
        landmarks_list.append(int(p.x - left))
        landmarks_list.append(int(p.y - top))
    return landmarks_list


if __name__ == '__main__':
    if not os.path.exists(FACE_DATA_DIR):
        os.makedirs(FACE_DATA_DIR)

    # initialize aligner
    aligner = ResizeAligner(desired_fh=H, desired_fw=W)

    # find all pngs in PAIN
    pngs = glob.glob(DATA_DIR + 'Images/*/*/*.png')
    data_csv = []

    # crop image and save
    # add tlbr, landmarks to csv
    for png in tqdm(pngs):
        png_name = os.path.basename(png)
        image = cv2.imread(png)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face, tlbr = crop_face(image)
        t, l, b, r = tlbr
        landmarks = get_landmarks(image_rgb, t, l, b, r)
        face, landmarks = aligner.align(image, dlib.rectangle(t,l,b,r), landmarks, face_img=face)

        # write image
        cv2.imwrite(FACE_DATA_DIR+png_name, face)

        # record csv info
        tlbr_str = [str(x) for x in [t, l, b, r]]
        landmarks_str = [str(x) for x in landmarks]
        data_csv.append([FACE_DATA_DIR+png_name, png, tlbr_str, landmarks_str, W, H])

    data_csv = pd.DataFrame(data_csv)
    data_csv.columns = ['img_path', 'orig_img_path', 'tlbr', 'landmarks', 'w', 'h']
    data_csv.to_csv(CSV_PATH, index=False)