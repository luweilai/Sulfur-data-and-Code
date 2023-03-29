import cv2
import numpy as np
import pandas as pd
import math
from google.colab.patches import cv2_imshow


img = cv2.imread('C002.tif')
ori_img = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(3, 3))
claheNorm = clahe.apply(gray)

blurred = cv2.GaussianBlur(claheNorm, (3, 3), 0)
th3 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV,127,64)
cv2_imshow(th3)
mask = th3.copy()
des = cv2.bitwise_not(mask)
contour,hier = cv2.findContours(des,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

contour = [c for c in contour if 1<cv2.contourArea(c) <500]
n_id = 0
diameter = []

for cnt in contour:
    cv2.drawContours(des,[cnt],0,255,-1)
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    diameter.append(radius)
    center = (int(x),int(y))
    #cv2.putText(ori_img,str(n_id),center,cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
    n_id +=1       
    radius = int(radius)
    
    cv2.circle(ori_img,center,radius,(0,0,255), 1, lineType=cv2.LINE_AA)
pd.DataFrame({"radius":diameter}).to_csv("radius.csv")
cv2.imwrite("radius.tif", ori_img)
cv2_imshow(ori_img)