import cv2
import numpy as np
import pickle
import joblib


forest = joblib.load(r"model1.pkl")

haar_data = cv2.CascadeClassifier('data.xml')
mask = {0 : "No Mask", 1 : "Mask"}
capture = cv2.VideoCapture(0)
while True:
    flag, img = capture.read()
    if flag:
        img = cv2.flip(img, 90)
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50, 50))
            face  = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = np.array(face).reshape(50 * 50 )
            inputs = [(np.asfarray(face.tolist()) / 255.0 * 0.99) + 0.01]
            prediction = forest.predict(inputs)
            img = cv2.putText(img, mask[prediction[0]], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.imshow("result", img)
        if cv2.waitKey(2) == 27:
            break
    else:
        break
capture.release()
cv2.destroyAllWindows()
