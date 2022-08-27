import numpy as np 
import cv2
import os   
import random 
import matplotlib.pyplot as plt

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def detect_face(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    faces = face_cascade.detectMultiScale(gray)
    
    (x, y, w, h) = faces[0]
    
    return gray[y:y+w, x:x+h], faces[0]

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def prepare_data(folder):
    x_train=[]
    y_labels=[]
    LABELS=[]
    index=0
    dirs = os.listdir(folder)
    for dir_name in dirs :
        index+=1
        abs_path_dir = folder + "/" + dir_name
        LABELS.append(dir_name)
        names = os.listdir(abs_path_dir)
        for img_name in names :
            img_path = abs_path_dir + "/" + img_name
            img = cv2.imread(img_path)
            cv2.waitKey(10)
            face, rect = detect_face(img)
            draw_rectangle(img,rect)
            cv2.imshow("Training on images", cv2.resize(img, (600, 400)))

            if face is not None :
                x_train.append(face)
                y_labels.append(index)
                         
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return x_train, y_labels, LABELS


def augment_data(imgs,rotations,labels):
    resulting_imgs=[]
    y_labels=[]
    for(img,label) in zip(imgs,labels):
        rows,cols=img.shape
        for j in rotations:
            rot_matrix = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),j,1)
            affined_img = cv2.warpAffine(img,rot_matrix,(cols,rows),borderValue=(1,1,1))
            resulting_imgs.append(affined_img)
            y_labels.append(label)
    return resulting_imgs, np.array(y_labels)


faces,y_labels,LABELS = prepare_data("dataset")
print("Total faces:", len(faces))

rotations = np.arange(0, 40)

X_augmented, Y = augment_data(faces, rotations, y_labels)

print('training ... ')
face_recognizer.train(X_augmented, Y)
face_recognizer.save('model.yml')

face_recognizer.read('model.yml')

for i in range(len(X_augmented)):
    cv2.imshow("augment", cv2.resize(X_augmented[i], (600, 400)))
    cv2.waitKey(1)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 0), 5)

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)

    label, confidence = face_recognizer.predict(face)
    print("predicted label:",label)
    prediction = LABELS[label-1]
    draw_rectangle(img, rect)
    draw_text(img, prediction, rect[0], rect[1]-5)
    return img

test_path ="dataset/Ghassane/" + os.listdir("dataset/Ghassane")[0]
test_img = cv2.imread(test_path)
prediction1 = predict(test_img)
cv2.imshow("test", cv2.resize(prediction1, (600, 400)))
cv2.waitKey(0)

if 0xFF == ord('q'):
    cv2.destroyAllWindows()
    cv2.waitKey(1)  
    cv2.destroyAllWindows()