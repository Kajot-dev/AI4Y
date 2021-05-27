print("Initializing libraries...")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import trainer as tr
import face_swap as fc
import dlib
import cv2
import json
import random
from tensorflow.keras.models import load_model
import numpy as np
face_detector = dlib.get_frontal_face_detector()
def ClassifyAIFaces(forceUpdate=False):
    if (not forceUpdate):
        try:
            with open('Face scrapper/labels.json', 'r') as fp:
                data = json.load(fp)
        except FileNotFoundError:
            with open('Face scrapper/labels.json', 'x') as fp:
                json.dump({}, fp)
            data = {}
    else:
        data = {}
    tab = os.listdir('./Face scrapper/faces')
    pred = load_model("trained.h5")
    filenames_list = []
    for element in tab:
        filename, ext = os.path.splitext(element)
        if (ext == ".jpg"):
            filenames_list.append(filename)
    print("Predicting...")
    int_list = list(map(int, filenames_list))
    int_list.sort()
    filenames_list = list(map(str, int_list))
    i = 0
    for name in filenames_list:
        if (name not in data.keys()):
            i += 1
            if (i % 100 == 0):
                with open('Face scrapper/labels.json', 'w+') as file:
                    json.dump(data, file)
            img = cv2.imread("Face scrapper/faces/"+name+".jpg")
            img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector(img_g)
            face = [faces[0].top(), faces[0].height(), faces[0].left(), faces[0].width()]
            crop = img[face[0]:face[0]+face[1], face[2]:face[2]+face[3]]
            #cv2.imshow("crop", crop)
            #cv2.waitKey(0)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pr, l = tr.Predict(model=pred, input_image=crop)
            data[name] = "".join(pr)
            print(name, ":",  l)
    with open('Face scrapper/labels.json', 'w+') as file:
        json.dump(data, file)
def AddImages(img_1, img_2):
    if (img_1.shape[0] > img_2.shape[0]):
        img_2 = cv2.resize(img_2, (img_1.shape[0], int(img_2.shape[1]*(img_1.shape[0]/img_2.shape[0]))))
    else:
        img_1 = cv2.resize(img_1, (img_2.shape[0], int(img_1.shape[1] * (img_2.shape[0] / img_1.shape[0]))))
    canvas = np.empty((img_1.shape[0], img_1.shape[1]+img_2.shape[1], 3), np.uint8)
    canvas[0:img_1.shape[0], 0:img_1.shape[1]] = img_1
    canvas[0:img_2.shape[0], img_1.shape[1]:img_1.shape[1]+img_2.shape[1]] = img_2
    return canvas, (img_1.shape, img_2.shape)
def Anonimize(path, name="anonimized"):
    print("Loading model...")
    pred = load_model("trained.h5")
    try:
        with open('Face scrapper/labels.json', 'r') as fp:
            labels = json.load(fp)
    except FileNotFoundError:
        print("ClassifyAIimages first!")
        return 1
    img = cv2.imread(path)
    org_shape = img.shape
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Detecting face and predicting...")
    faces = face_detector(img_g)
    face = [faces[0].top(), faces[0].height(), faces[0].left(), faces[0].width()]
    crop = img[face[0]:face[0] + face[1], face[2]:face[2] + face[3]]
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pr, l = tr.Predict(model=pred, input_image=crop_rgb)
    print("PREDICTED LABELS :",", ".join(l))
    print("Choosing AI picture...")
    pr_key = "".join(pr)
    valid_imgs = list(filter(lambda x: labels[x] == pr_key, list(labels.keys())))
    if (len(valid_imgs) == 0):
        print("Download more AI images (use Face scrapper) or choose another face")
        return
    AI_img = random.choice(valid_imgs)
    AI_img = cv2.imread("Face scrapper/faces/"+AI_img+".jpg")
    print("Merging images...")
    merged, shapes = AddImages(img, AI_img)
    print("Swapping faces...")
    out = fc.SwapFaces(merged)
    final = out[0:shapes[0][0], 0:shapes[0][1]]
    final = cv2.resize(final, (org_shape[1], org_shape[0]))
    print("Saving as " + name + ".jpg...")
    cv2.imwrite(name+".jpg", final)
    print("Anonimization finished!")
    return final
if (__name__ == "__main__"):
    Anonimize("2.jpg", "GAW_anonimized")