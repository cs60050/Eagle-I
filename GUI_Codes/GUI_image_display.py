from Tkinter import *
import tkMessageBox
import tkFileDialog
import os
import glob
import cv2
import tkFont

top = Tk()
def display_img(img_name):
	print "img_name: is %s" %img_name 
        img = cv2.imread(img_name)
	screen_res = 1280, 720
	scale_width = screen_res[0] / img.shape[1]
	scale_height = screen_res[1] / img.shape[0]
	scale = min(scale_width, scale_height)
	window_width = int(img.shape[1] * scale)
	window_height = int(img.shape[0] * scale)

	cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('dst_rt', window_width, window_height)

	cv2.imshow('dst_rt', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def helloCallBack():
   	file_path_string = tkFileDialog.askopenfilename()
        display_img(file_path_string)
var = StringVar()
label = Label( top, textvariable=var, relief=RAISED,bg= "green" ,width = "100",height = "20",borderwidth= "20",fg= "pink")


var.set("Machine Learning Term Project \n\n Sample GUI For Image Display \n\n Team - Eagle I")

B = Button(top, text ="Select the image u want to display from Ur directory", command = helloCallBack,bg ="yellow",width = "50",height = "5",borderwidth= "20",fg = "blue",activebackground = "blue",activeforeground = "black")
label.pack()
B.pack()
top.mainloop()
