import os
import cv2
import dlib
import numpy as np
# use cuda device
if (dlib.cuda.get_num_devices() > 0):
    dlib.cuda.set_device(0)
if (dlib.DLIB_USE_CUDA):
    print("Dlib using gpu!")
print("Loading face detectors to memory...")
face_detector = dlib.get_frontal_face_detector()
face_landmarks_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def TakePicture():
    # try webcam
    cam = cv2.VideoCapture(0)
    s, img = cam.read()
    if (s):
        SwapAllFaces(img)
        return True
    else:
        # try usb cam
        cam = cv2.VideoCapture(1)
        s, img = cam.read()
        if (s):
            SwapAllFaces(img)
            return True
        else:
            print("Could not take picture from camera!")
            return False

def SwapFromFile(filepath):
    # determine project path
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, filepath)
    path, filename = os.path.split(filepath)
    if (os.path.isfile(filepath)):
        img = cv2.imread(filepath)
        SwapAllFaces(img, filename.split("."))
    else:
        print("File doesn't exist!")
def AnalyzeFaces(img, reqF=2):
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rotations = []  # rotation of each face
    landmarks = []
    print("Detecting faces...")
    faces = face_detector(img_g)
    # 45 degree line
    line = dlib.line(dlib.point(0, 0), dlib.point(20, 20))
    # check number of faces
    if (len(faces) < reqF):
        print('Image should have at least '+str(reqF)+' face(s)!')
        return 1
    print("Detected faces: " + str(len(faces)))
    face_nr = 0
    for face in faces:
        face_nr += 1
        print("Analyzing face " + str(face_nr) + ":")
        landmarks_v = face_landmarks_predictor(img_g, face)
        landmarks.append(landmarks_v)
        print("Calculating rotation for face " + str(face_nr) + "...")
        rotations.append(dlib.angle_between_lines(line, dlib.line(landmarks_v.part(27), landmarks_v.part(8))) - 45)
    return (landmarks, rotations)
def GetShapes(img, landmarks_a):
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped_faces_a = []
    cropped_faces = []
    masks_a = []
    masks = []
    bounding_boxes = []
    important_points = [48, 59, 58, 57, 56, 55, 54, 15, 16, 26, 25, 24, 23, 22, 27, 21, 20, 19, 18, 17, 0, 1]
    for landmarks in landmarks_a:
        l_points_a = []
        l_points = []
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            l_points_a.append((x, y))
        for i in important_points:
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            l_points.append((x, y))
        # crop full face
        f_points_a = np.array(l_points_a, np.int32)
        convexhull_a = cv2.convexHull(f_points_a)
        mask_a = np.zeros_like(img_g)
        cv2.fillConvexPoly(mask_a, convexhull_a, 255)
        face_now_a = cv2.bitwise_and(img, img, mask=mask_a)
        bounds = cv2.boundingRect(mask_a)
        cr_face_a = face_now_a[bounds[1]:bounds[1] + bounds[3], bounds[0]:bounds[0] + bounds[2]]
        cr_mask_a = mask_a[bounds[1]:bounds[1] + bounds[3], bounds[0]:bounds[0] + bounds[2]]
        # crop important part
        f_points = np.array(l_points, np.int32)
        convexhull = cv2.convexHull(f_points)
        mask = np.zeros_like(img_g)
        cv2.fillConvexPoly(mask, convexhull, 255)
        face_now = cv2.bitwise_and(img, img, mask=mask)
        cr_face = face_now[bounds[1]:bounds[1] + bounds[3], bounds[0]:bounds[0] + bounds[2]]
        cr_mask = mask[bounds[1]:bounds[1] + bounds[3], bounds[0]:bounds[0] + bounds[2]]
        # save
        cropped_faces_a.append(cr_face_a)
        cropped_faces.append(cr_face)
        masks_a.append(cr_mask_a)
        masks.append(cr_mask)
        bounding_boxes.append(bounds)
    return ((cropped_faces_a, cropped_faces), (masks_a, masks), bounding_boxes)
def BlurFaces(img, masks_a, bounding_boxes, faces):
    print("Blurring faces...")
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurPower = int(((img_g.shape[0]+img_g.shape[1])/2)*0.25)
    if (blurPower % 2 == 0):
        blurPower += 1
    empty = np.zeros_like(img)
    output = img.copy()
    (h, w, x) = img.shape
    T = np.float32([[1, 0, 0], [0, 1, int(h * -0.018)]])
    for i in range(len(masks_a)):
        final_mask_a = np.zeros_like(img_g)
        final_mask_a[bounding_boxes[i][1]:bounding_boxes[i][1] + bounding_boxes[i][3], bounding_boxes[i][0]:bounding_boxes[i][0] + bounding_boxes[i][2]] = masks_a[i]
        final_mask_a = cv2.warpAffine(final_mask_a, T, (w, h))
        inv_mask_a = cv2.bitwise_not(final_mask_a)
        img_b_a = cv2.bitwise_and(img, img, mask=inv_mask_a)
        empty[bounding_boxes[i][1]:bounding_boxes[i][1] + bounding_boxes[i][3],bounding_boxes[i][0]:bounding_boxes[i][0] + bounding_boxes[i][2]] = faces[i]
        empty = cv2.blur(empty, (blurPower, blurPower), cv2.BORDER_DEFAULT)
        empty = cv2.bitwise_and(empty, empty, mask=final_mask_a)
        img_b_a = cv2.add(img_b_a, empty);
        center = (bounding_boxes[i][0] + int(bounding_boxes[i][2] / 2), bounding_boxes[i][1] + int(bounding_boxes[i][3] / 2))
        center = (center[0], center[1] + int(h * -0.018))
        output = cv2.seamlessClone(img_b_a, output, final_mask_a, center, cv2.NORMAL_CLONE)
    return output
def ResizeFaces(cropped_f, masks, bounding_boxes, rotations):
    r_masks_a = []
    r_faces_a = []
    for i in range(len(cropped_f[0])):
        if i >= len(cropped_f[0]) - 1:
            y = 0
        else:
            y = i + 1
        dim = (bounding_boxes[y][2], bounding_boxes[y][3])
        r_faces_a.append(cv2.resize(cropped_f[0][i], dim))
        r_masks_a.append(cv2.resize(masks[0][i], dim))
    box = bounding_boxes.pop(0)
    bounding_boxes.append(box)
    rot = rotations.pop(0)
    rotations.append(rot)
    return (r_faces_a, r_masks_a, bounding_boxes, rotations)


def ApplyFaces(img, faces, masks, bounding_boxes, rotations):
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = img.copy()
    face_nr = 0
    for i in range(len(masks)):
        face_nr += 1
        print("Positioning and applying a face "+str(face_nr)+" to the final image...")
        center = (bounding_boxes[i][0] + int(bounding_boxes[i][2] / 2), bounding_boxes[i][1] + int(bounding_boxes[i][3] / 2))
        final_mask_a = np.zeros_like(img_g)
        final_mask_a[bounding_boxes[i][1]:bounding_boxes[i][1]+bounding_boxes[i][3], bounding_boxes[i][0]:bounding_boxes[i][0]+bounding_boxes[i][2]] = masks[i]
        (h, w) = final_mask_a.shape
        r = cv2.getRotationMatrix2D(center, rotations[i-1]-rotations[i], 1.0)
        final_mask_a = cv2.warpAffine(final_mask_a, r, (w, h))
        inv_mask_a = cv2.bitwise_not(final_mask_a)
        img_b_a = cv2.bitwise_and(output, output, mask=inv_mask_a)
        empty_a = np.zeros_like(output)
        empty_a[bounding_boxes[i][1]:bounding_boxes[i][1] + bounding_boxes[i][3], bounding_boxes[i][0]:bounding_boxes[i][0] + bounding_boxes[i][2]] = faces[i]
        empty_a = cv2.warpAffine(empty_a, r, (w, h))
        img_b_a = cv2.add(img_b_a, empty_a)
        print("Recoloring face "+str(face_nr)+"...")
        output = cv2.seamlessClone(img_b_a, output, final_mask_a, center, cv2.MIXED_CLONE)
    return output
def SwapAllFaces(img, filename="swapped.jpg"):
    landmarks, rotations = AnalyzeFaces(img)
    faces, masks, bounds = GetShapes(img, landmarks)
    img_bl = BlurFaces(img, masks[1], bounds, faces[1])
    faces, masks, bounding_boxes, rotations = ResizeFaces(faces, masks, bounds, rotations)
    output = ApplyFaces(img_bl, faces, masks, bounding_boxes, rotations)
    extension = filename.pop()
    s = ""
    name = s.join(filename)+"_swapped."+extension
    print("Saving as "+name+"...")
    cv2.imwrite(name, output)
    print("Done!")
def SwapFaces(img):
    landmarks, rotations = AnalyzeFaces(img)
    faces, masks, bounds = GetShapes(img, landmarks)
    img_bl = BlurFaces(img, masks[1], bounds, faces[1])
    faces, masks, bounding_boxes, rotations = ResizeFaces(faces, masks, bounds, rotations)
    output = ApplyFaces(img_bl, faces, masks, bounding_boxes, rotations)
    return output
def Choice():
    print("1 = take picture, 2 = open from file, else = close")
    char = input()
    if (char == '1'):
        TakePicture()
    elif (char == '2'):
        SwapFromFile(input("Podaj scieżkę do zdjęcia: "))
    else:
        return 0

if (__name__ == "__main__"):
    Choice()
