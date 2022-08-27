import cv2

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('model.yml')
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
LABELS=['giannis','amine','Ghassane']

cap=cv2.VideoCapture(0)

while True:
    ref,frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:

        box=gray[y:y+h,x:x+w]
        label,accuracy=face_recognizer.predict(box)
        prediction = LABELS[label-1]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, prediction, (x, y), cv2.FONT_HERSHEY_PLAIN, 6, (0, 255, 255), 5)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
