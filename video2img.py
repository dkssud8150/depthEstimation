import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("gangnam2.avi")



i = 0
while True:
    ok, frame = cap.read()
    if not ok : break

    os.makedirs("data/img/acelab", exist_ok=True)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print(frame.shape)

    savePath = "data/img/acelab/" + str(i).zfill(5) + ".png"
    cv2.imwrite(savePath, frame)

    i+=1