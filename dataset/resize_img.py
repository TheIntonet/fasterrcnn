import cv2
import os

DATADIR = "images"
DESTDIR = "destimges"

sizex = 224
sizey = 224
dim = (sizey, sizex)

path = os.path.join(DATADIR)
for img in os.listdir(path):
    data = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
    res = cv2.resize(data, (sizex, sizey))
    cv2.imwrite(os.path.join(os.path.join(DESTDIR), img), res)
print("Saved..")
