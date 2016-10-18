import cv2 
import numpy as np
import sys
import time

#-------------Thresholding Image--------------#

src_img = cv2.imread('./data/img_2.jpg', 1)
grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

gud_img = cv2.adaptiveThreshold(grey_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,101,2)
# ret3,thr_img = cv2.threshold(gud_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
closing = cv2.morphologyEx(gud_img, cv2.MORPH_CLOSE, kernel, iterations = 2) # To remove "pepper-noise"

kernel1 = np.array([[1,0,1],[0,1,0],[1,0,1]], dtype = np.uint8)
final = cv2.erode(closing,kernel1,iterations = 1)

# blur = cv2.bilateralFilter(erosion,9,75,75)


#-------------Displaying Image----------------#

cv2.namedWindow('Source Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Threshold Image', cv2.WINDOW_NORMAL)

cv2.imshow("Source Image", src_img)
cv2.imshow("Threshold Image", final)


#-------------Closing Windows-----------------#

k = cv2.waitKey(0)
if k & 0xFF == ord('s'):
	comment = input("Comment:-\n ")
	cv2.imwrite('./data/test_result/'+comment+'.jpg',final)
	# cv2.imwrite('OKAY.jpg',final)
	print("Completed")
else:
	cv2.destroyAllWindows()
