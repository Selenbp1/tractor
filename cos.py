import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5 import QtGui
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree;
import sys
from PIL import Image
import matplotlib.pylab as plt;
import matplotlib.pyplot as plt;

isDragging = False;
x0, y0, w, h = -1, -1, -1, -1
blue, red = (255, 0, 0), (0, 0, 255)

xmlpath = '항공영상/01. 월동무/학습데이터/xml/'
imgpath = '항공영상/01. 월동무/학습데이터/image/'
filename = '2011A0025_0206'
imgetc = '.jpg'
xmletc ='.xml'

def check(RGB):
    if RGB == "RGB":
        print("현재 수확 시기입니다. 해당 밭 면적에 맞는 농기계")

def tractor(img, x, y):
    im = im = Image.open(img)
    tractor = Image.open("imeage/tractor.png")
    im.paste(tractor, (x,y));
    
    im.save('cropped2.png')
    

def chooserecommend(weight, plant):
    if 1000<weight<20000000 and plant == "월동무":
        return 512
        

def closeimg(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.destroyAllWindows()
        createPyQt5()
        
def createPyQt5():
    app = QtWidgets.QApplication([])
    label = QtWidgets.QLabel()

    img_array = np.fromfile('cropped.png', np.uint8)
    img = cv2. imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2. cvtColor(img, cv2.COLOR_BGR2RGB) 

    h,w,c = img.shape;
    qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap.fromImage(qImg)  
    label.setPixmap(pixmap)
    label.resize(pixmap.width(), pixmap.height())
    label.show()
    
    tractor('cropped.png', 32, 30)
    img_array = np.fromfile('cropped2.png', np.uint8)
    img = cv2. imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2. cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h,w,c = img.shape;
    qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
    pixmap = QtGui.QPixmap.fromImage(qImg)  
    label.setPixmap(pixmap)
    label.resize(pixmap.width(), pixmap.height())
    label.show()

    app.exec_()


def onMouse(event, x, y, flags, param):
    global isDragging, x0, y0, img
    if event == cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        x0 = x
        y0 = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            img_draw = img.copy()
            cv2.rectangle(img_draw, (x0, y0), (x, y), blue, 2)
            cv2.imshow('img', img_draw)
    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            w = x - x0
            h = y - y0
            if w > 0 and h > 0:
                img_draw = img.copy()
                cv2.rectangle(img_draw, (x0, y0), (x, y), red, 2)
            
                cv2.imshow('img', img_draw)
                (x,y), (w,h) = (512,512), (512,512)
                roi = img[y0:y0+h, x0:x0+w]                
                
                cv2.imshow('cropped', roi)
                cv2.moveWindow('cropped', 0, 0)
                cv2.imwrite('./cropped.png', roi)
                cv2.destroyWindow("img")
            
                x,y,z = roi.shape
                weight = x*y*z            
                doc = ET.parse(xmlpath+filename+xmletc)
                root = doc.getroot()
                plant = str(root.find('CTVT_CROPS_NM').text);
                
                print("현재 심어진 작물은:" + plant + "선택된 밭의 크기는:", weight);
                
                RGB = 'RGB'
                check(RGB)
                                
                print(chooserecommend(weight, plant))
                
                cv2.setMouseCallback('cropped', closeimg)
                
            else:
                print('drag should start from left-top side')
                
                
img_array = np.fromfile(imgpath+filename+imgetc, np.uint8)

img = cv2. imdecode(img_array, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse)
cv2.waitKey()
cv2.destroyAllWindows()