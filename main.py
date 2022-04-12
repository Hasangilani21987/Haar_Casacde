# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1 Cascade File

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_eye_tree_eyeglasses.xml')
#
# img = cv2.imread("photo.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(100, 100))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()


# 2nd Cacade File

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_eye.xml')
#
# img = cv2.imread("photo.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(70, 70))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()



# 3rd Cacade File

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_frontalcatface.xml')
#
# img = cv2.imread("photo.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(40, 40))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()


# 4th Cacade File

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_frontalcatface_extended.xml')
#
# img = cv2.imread("photo.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(70, 70))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()


# 5th Cacade File

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_frontalface_alt.xml')
#
# img = cv2.imread("photo.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(70, 70))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()



# 6th Cacade File

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_frontalface_alt2.xml')
#
# img = cv2.imread("photo.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(70, 70))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()


# 7th Cacade File

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_frontalface_alt_tree.xml')
#
# img = cv2.imread("photo.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(70, 70))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# 8th Cacade File

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_frontalface_default.xml')
#
# img = cv2.imread("photo.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(70, 70))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()


# 9th Cacade File

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_lefteye_2splits.xml')
#
# img = cv2.imread("photo.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(100, 100))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()


# 10th Cacade File

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_fullbody.xml')
#
# img = cv2.imread("full_body.png")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(80, 120))
#
# print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 4)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# 11th Cacade File

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_lowerbody.xml')
#
# img = cv2.imread("full_body.png")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(80, 120))
#
# print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 4)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# 12th Cacade File

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_profileface.xml')
#
# img = cv2.imread("photo.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(80, 120))
#
# print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 4)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# 13th Cascade Files

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_righteye_2splits.xml')
#
# img = cv2.imread("photo.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(95, 95))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 4)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# 14th Cascade Files

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_upperbody.xml')
#
# img = cv2.imread("full_body.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(50, 50))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 4)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# 14th Cascade Files

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_smile.xml')
#
# img = cv2.imread("simile.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(50, 50))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 4)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# 15th Cascade Files
#
# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_smile1.xml')
#
# img = cv2.imread("simile.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(50, 50))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 4)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

#  16th Cascade Files

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_russian_plate_number.xml')
#
# img = cv2.imread("plate.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(50, 50))
#
# print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 4)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()


#  17th Cascade Files

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_wallclock.xml')
#
# img = cv2.imread("clock.jpg")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(100, 100))
#
# # print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 4)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

#  18th Cascade Files

# classifier = cv2.CascadeClassifier('haar_xmlFiles/haarcascade_licence_plate_rus_16stages.xml')
#
# img = cv2.imread("license_plate.png")
# resized = cv2.resize(img, (400, 400))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#
#
# faces = classifier.detectMultiScale(gray, minSize=(100, 100))
#
# print(faces)
#
# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 4)
#
# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()



# from icrawler.builtin import BingImageCrawler

# we are building cats image detection that's why we put cat here
# if you want some other images then put that name in classes list
# classes = ['cats images']
# number = 100
# here root directory is find your root directory there u will find
# new file name data in which all images are saved.
# for c in classes:
#     bing_crawler = BingImageCrawler(storage={'root_dir': f'n/{c.replace(" ",".")}'})
#     bing_crawler.crawl(keyword=c, filters=None, max_num=number, offset=0)

# Custom Cascader Train Model

# classifier = cv2.CascadeClassifier('cascade.xml')
#
# img = cv2.imread("cat.jpg")
# resized = cv2.resize(img, (400, 300))
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# faces = classifier.detectMultiScale(gray, minSize=(150, 150))

# print(faces)

# if len(faces) != 0:
#     for (x, y, w, h) in faces:
#         cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv2.imshow("Face Detection", gray)
# cv2.waitKey(0)

# cv2.destroyAllWindows()













