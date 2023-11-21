from detect import run
import cv2

image = cv2.imread("IMG11.jpg")
pt = open("best.pt")
data = open("custom_dataset.yaml")

run(pt, image, data)