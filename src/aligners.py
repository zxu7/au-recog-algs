import cv2
import numpy as np
from .utils import points_to_np

# dlib facial keypoints
# LEFT_EYE_INDICES = list(range(36, 42))
# RIGHT_EYE_INDICES = list(range(42, 48))
LEFT_EYE_INDICES = list(range(42, 48))
RIGHT_EYE_INDICES = list(range(36, 42))


class Aligner(object):
    def __init__(self, **kwargs):
        pass

    def align(self, **kwargs):
        pass


class ResizeAligner(Aligner):
    """align face by size"""
    def __init__(self, desired_fw=128, desired_fh=128):
        super(ResizeAligner, self).__init__()
        self.desired_fw = desired_fw
        self.desired_fh = desired_fh

    def align(self, image, rect, keypoints, **kwargs):
        """
        :param image: numpy.array; bgr
        :param rect: dlib.rectangle;
        :param keypoints: list; e.g. [x0, y0, ... , x67, y67]
        :return:
        """
        h, w = image.shape[:2]
        if 'face_img' in kwargs:
            face_img = kwargs['face_img']
        else:
            # obtain face_img from image, rect
            pass

        aln_image = cv2.resize(face_img, (self.desired_fw, self.desired_fh))
        aln_keypoints = np.array(keypoints).reshape((-1, 2)) * np.array([self.desired_fw/w, self.desired_fh/h])
        aln_keypoints = aln_keypoints.flatten().tolist()
        return aln_image, aln_keypoints


class FaceAligner(object):
    """align face by rotating the l/r eyes center of mass angle and scale l/r eyes dist"""

    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
    # TODO: need to align and return keypoints
    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(image, rect)
        points = shape.parts()
        shape = points_to_np(points)

        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = LEFT_EYE_INDICES[0], LEFT_EYE_INDICES[-1]
        (rStart, rEnd) = RIGHT_EYE_INDICES[0], RIGHT_EYE_INDICES[-1]
        leftEyePts = shape[lStart:lEnd + 1]
        rightEyePts = shape[rStart:rEnd + 1]
        print(leftEyePts)
        print(rightEyePts)
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                      (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output