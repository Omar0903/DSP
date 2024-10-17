# Importing all Needed Libraries
from tkinter import *
from tkinter import ttk
from tkinter import Tk
from tkinter import filedialog
from tkinter import   filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from function import *

MainScreen = Tk()
MainScreen.geometry('1280x720')
MainScreen.resizable(False , False)
MainScreen.title('DSP Task')
MainScreen.iconbitmap("DSP.ico")
MainScreen.config(background='navajowhite')
fr1  = Frame(width='1200',height='720',bg='navajowhite')
fr2  = Frame(fr1,width='350',height='280',bg='navajowhite')
fr3  = Frame(fr1,width='300',height='280',bg='navajowhite')
plot = Frame(width='1900',height='400',bg='navajowhite')
Lb1  = Label(text=' Wave type ',fg='black',bg='white',font=25,width=25)
Lb2  = Label(fr2,text=' Amplitude ',fg='black',bg='white',font=25,width=25)
Lb3  = Label(fr2,text=' AnalogFrequency ',fg='black',bg='white',font=25,width=25)
Lb4  = Label(fr2,text=' SamplingFrequency',fg='black',bg='white',font=25,width=25)
Lb5  = Label(fr2,text=' PhaseShift ',fg='black',bg='white',font=25,width=25)
Lb7  = Label(fr3,text=' Inputs ',fg='black',bg='white',font=25,width=20)
Lb8  = Label(fr2,text=' Signal Information ',fg='black',bg='white',font=25,width=25)
Lb9  = Label(fr1,text=' Welcome TO Signal Digital Program ',fg='black',bg='white',font=25,width=35)
Lb10 = Label(text=' Normalization range ',fg='black',bg='white',font=25,width=25)
bt1  = Button(MainScreen,text='Generate',fg='black',bg='white',width=15,height=2,command=lambda: GenerateSignal(En1, En2, En3, En4, cmbo1, plot))  
bt2  = Button(MainScreen,text='Open Folder',fg='black',bg='white',command=ReadFile,width=15,height=2)
bt3  = Button(MainScreen,text='Addition',fg='black',bg='white',width=15,height=2,command=ChooseFileForAddition)  
bt4  = Button(MainScreen,text='Subtraction',fg='black',bg='white',width=15,height=2,command=ChooseFileForSubtraction)  
bt5  = Button(MainScreen,text='Multiplication',fg='black',bg='white',width=15,height=2,command=GenerateSignal)  
bt6  = Button(MainScreen,text='Squaring',fg='black',bg='white',width=15,height=2,command=GenerateSignal)  
bt7  = Button(MainScreen,text='Normalization',fg='black',bg='white',width=15,height=2,command=ChooseFileForNormalization)  
bt8  = Button(MainScreen,text='Accumulation ',fg='black',bg='white',width=15,height=2,command=GenerateSignal)  





En1 = Entry(fr3,fg='black',bg='white',font= 15,justify="center")
En2 = Entry(fr3,fg='black',bg='white',font= 15,justify="center")
En3 = Entry(fr3,fg='black',bg='white',font= 15,justify="center")
En4 = Entry(fr3,fg='black',bg='white',font= 15,justify="center")
bt1.place(x=50,y=400)
bt2.place(x=200,y=400)
bt3.place(x=350,y=400)
bt4.place(x=500,y=400)
bt5.place(x=650,y=400)
bt6.place(x=800,y=400)
bt7.place(x=950,y=400)
bt8.place(x=1100,y=400)

fr1.place(x=5,y=5)
fr2.place(x=30,y=80)
fr3.place(x=400,y=80 )
plot.place(x=5,y=420)
En1.place(x=40,y=60)
En2.place(x=40,y=100)
En3.place(x=40,y=140)
En4.place(x=40,y=180)
Lb1.place(x=80,y=310)
Lb2.place(x=40,y=60)
Lb3.place(x=40,y=100)
Lb4.place(x=40,y=140)
Lb5.place(x=40,y=180)
Lb7.place(x=40,y=20)
Lb8.place(x=40,y=20)
Lb9.place(x=400,y=30)
Lb10.place(x=80,y=350)
cmbo1 = ttk.Combobox(MainScreen, value = ('Sin','Cos'),background ='silver')
cmbo2 = ttk.Combobox(MainScreen, value = ('from 0 to 1','from -1 to 1'),background ='silver')

cmbo1.set('Sin')
cmbo2.set('from 0 to 1')
cmbo1.place(x=350,y=310,height=23,width=50)
cmbo2.place(x=350,y=350,height=23,width=80)
MainScreen.mainloop()



