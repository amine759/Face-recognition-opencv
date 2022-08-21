import numpy as np 
import cv2
import os 

labels={'amine':1,'Ghassane':2,'giannis':3}
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
    dirs = os.listdir(folder)
    for dir_name in dirs :
        abs_path_dir = folder + "/" + dir_name
        names = os.listdir(abs_path_dir)
        for img_name in names :
            img_path = abs_path_dir + "/" + img_name
            img = cv2.imread(img_path)
            cv2.imshow("Training on image...", cv2.resize(img, (600, 500)))
            cv2.waitKey(100)
            face, rect = detect_face(img)
            draw_rectangle(img,rect)

            if face is not None :
               x_train.append(face)
               y_labels.append(labels[dir_name])
    return x_train, y_labels

faces,y_labels = prepare_data("dataset")
print("Total faces:", len(faces))

face_recognizer.train(faces, np.array(y_labels))
#recognizor.save('model.yml')



def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)

    label, confidence = face_recognizer.predict(face)
    for key,value in labels.items() :  
        if value == label :
            label_text = key

    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    return img

test_path ="dataset/amine/" + os.listdir("dataset/amine")[0]

test_img = cv2.imread(test_path)
prediction1 = predict(test_img)
cv2.imshow("test", cv2.resize(prediction1, (600, 500)))