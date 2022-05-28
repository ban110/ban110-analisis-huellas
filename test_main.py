import requests
import base64
import io
from PIL import Image


req = requests.post("http://127.0.0.1:5001/", files={'file': open('image.jpg', mode='rb')})
print(req.status_code)
# encode, distance
print(req.json()['distance'], "lines")
encoded_image = req.json()['encode'][2:-1]
msg = base64.b64decode(encoded_image)
buf = io.BytesIO(msg)
img = Image.open(buf)
img.save("decoded.jpg")

# import cv2 as cv
#
# img = cv.imread("received/16.jpg")
# ancho = img.shape[1]
# alto = img.shape[0]
# M = cv.getRotationMatrix2D((ancho//2, alto//2), 15, 1)
# img = cv.warpAffine(img, M, (ancho, alto))
# # cv.rotate(img, 30)
# cv.imwrite("salida.jpg", img)