from Tkinter import *
import tkMessageBox
import tkFileDialog
import os
import glob
import tkFont
top = Tk()

 ### when button clicks,it selects the file 
def helloCallBack():
   	file_path_string = tkFileDialog.askopenfilename()  #stores the file path in a variable
        os.system("python3 main.py "+str(file_path_string) ) #passes the file path(img path) through command line argument
        
var = StringVar()
label = Label( top, textvariable=var, relief=RAISED,bg= "green" ,width = "100",height = "20",borderwidth= "20",fg= "pink")

mb=  Menubutton ( top, text="menu", relief=RAISED,padx=3 )

var.set("Machine Learning Term Project \n\n GUI For Image Segmantation \n\n Team - Eagle I")

B = Button(top, text ="Select the image u want to Segment from Ur directory", command = helloCallBack,bg ="yellow",width = "50",height = "2",borderwidth= "20",fg = "blue",activebackground = "blue",activeforeground = "black")
mb.pack()
label.pack()
B.pack()
top.mainloop()
