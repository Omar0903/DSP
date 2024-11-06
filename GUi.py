# import Libraries
import tkinter as tk
from tkinter import Button, Label, Entry
from tkinter import *
from tkinter import ttk
from tkinter import Tk
from tkinter import   filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from function import *
from PIL import Image, ImageTk
import cv2
from test3 import *

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Multi-Page Example")
        # self.background_image = Image.open("j.jpg")  
        # self.background_photo = ImageTk.PhotoImage(self.background_image)

      # Initialize the first page
        self.page = 0
     
        self.create_page()
    def create_page(self):
        # Clear previous widgets
        for widget in self.master.winfo_children():
            widget.destroy()

        # Create content based on the current page
        if self.page == 0:
            self.CreatePageZero()
        elif self.page == 1:
            self.CreatePageOne()
        elif self.page == 2:
            self.CreatePageTwo()
        elif self.page == 3:
            self.CreatePageThree()
        elif self.page == 4:
            self.CreatePageFour()
        elif self.page == 5:
            self.CreatePageFive()
        elif self.page == 6:
            self.CreatePageSix()
        elif self.page == 7:
            self.CreatePageSeven()


        # Create Back and Next buttons for all pages
        back_button = Button(self.master, text="Back", command=self.back, bg="white", width=20)
        back_button.place(x=30, y=680)

        next_button = Button(self.master, text="Next", command=self.next, bg="white", width=20)
        next_button.place(x=1100, y=680)


    def CreatePageZero(self):
        label1 = Label(self.master, text="Welcome to digital signal proccessing project",font=("Arial", 25), background='white', width=50, justify="center",)
        label1.place(x=175, y=50)
      
    def CreatePageOne(self):
       

        label = Label(self.master, text="Generation Wave Signals and plot it", font=25, background='white', width=50, justify="center")
        label.pack(pady=20)

        fr1 = Frame(self.master, width='1280', height='720', bg='navajowhite')
        fr1.pack(fill='both', expand=True)
        fr2 = Frame(fr1, width='350', height='290', bg='white')
        fr2.place(x=30, y=50)
        fr3 = Frame(fr1, width='300', height='290', bg='white')
        fr3.place(x=500, y=50)

        Lb1 = Label(fr2, text=' Wave type ', fg='black', bg='lightgrey', font=25, width=25)
        Lb2 = Label(fr2, text=' Amplitude ', fg='black', bg='lightgrey', font=25, width=25)
        Lb3 = Label(fr2, text=' AnalogFrequency ', fg='black', bg='lightgrey', font=25, width=25)
        Lb4 = Label(fr2, text=' SamplingFrequency', fg='black', bg='lightgrey', font=25, width=25)
        Lb5 = Label(fr2, text=' PhaseShift ', fg='black', bg='lightgrey', font=25, width=25)
        Lb7 = Label(fr3, text=' Inputs ', fg='black', bg='lightgrey', font=25, width=20)
        Lb8 = Label(fr2, text=' Signal Information ', fg='black', bg='lightgrey', font=25, width=25)

        bt1 = Button(fr1, text='Generate', fg='black', bg='white', width=25, height=2, command=lambda: GenerateSignal(En1, En2, En3, En4, cmbo1))  
        bt2 = Button(fr1, text='Open Folder', fg='black', bg='white', command=ReadFile, width=25, height=2)

        bt1.place(x=850, y=360)
        bt2.place(x=1050, y=360)

        Lb1.place(x=40, y=220)
        Lb2.place(x=40, y=60)
        Lb3.place(x=40, y=100)
        Lb4.place(x=40, y=140)
        Lb5.place(x=40, y=180)
        Lb7.place(x=40, y=20)
        Lb8.place(x=40, y=20)

        En1 = Entry(fr3, fg='black', bg='lightgrey', font=15, justify="center", width=20)
        En2 = Entry(fr3, fg='black', bg='lightgrey', font=15, justify="center", width=20)
        En3 = Entry(fr3, fg='black', bg='lightgrey', font=15, justify="center", width=20)
        En4 = Entry(fr3, fg='black', bg='lightgrey', font=15, justify="center", width=20)

        En1.place(x=40, y=60)
        En2.place(x=40, y=100)
        En3.place(x=40, y=140)
        En4.place(x=40, y=180)
        cmbo1 = ttk.Combobox(fr3, value=('Sin', 'Cos'), background='silver')
        cmbo1.set('Sin')
        cmbo1.place(x=40, y=220, height=28, width=50)

    def CreatePageTwo(self):
        label = Label(self.master, text="Operations on signals", font=25, bg='white',width=30,justify="center")
        label.pack(pady=20)

        bt3  = Button(MainScreen,text='Addition',fg='black',bg='white',width=20,height=2,command=ChooseFileForAddition)  
        bt4  = Button(MainScreen,text='Subtraction',fg='black',bg='white',width=20,height=2,command=ChooseFileForSubtraction,)  
        bt5  = Button(MainScreen,text='Multiplication',fg='black',bg='white',width=20,height=2,command=lambda: Multiplication(En5))  
        bt6  = Button(MainScreen,text='Squaring',fg='black',bg='white',width=20,height=2,command=Squaring)  
        bt7  = Button(MainScreen,text='Normalization',fg='black',bg='white',width=20,height=2,command=lambda : ChooseFileForNormalization(cmbo2))  
        bt8  = Button(MainScreen,text='Accumulation ',fg='black',bg='white',width=20,height=2,command=ChooseFileForAccumulation)  
        bt9  = Button(MainScreen,text='Compare Signals ',fg='black',bg='white',width=20,height=2,command=CheckSamples) 

        bt3.place(x=260,y=470)
        bt4.place(x=420,y=470)
        bt5.place(x=580,y=470)
        bt6.place(x=740,y=470)
        bt7.place(x=900,y=470)
        bt8.place(x=1060,y=470)
        bt9.place(x=100,y=470)

        cmbo2 = ttk.Combobox(MainScreen, value = ('from 0 to 1','from -1 to 1'),background ='silver')
        cmbo2.set('from 0 to 1')
        cmbo2.place(x=445,y=350,height=23,width=80)

        Lb10 = Label(text=' Normalization range ',fg='black',bg='white',font=25,width=25)
        Lb11 = Label(text=' Multiplication constant ',fg='black',bg='white',font=25,width=25)

        Lb10.place(x=75,y=350)
        Lb11.place(x=75,y=390)

        En5 = Entry(MainScreen,fg='black',bg='white',font= 15,justify="center",width=10)
        En5.place(x=445,y=390)


        

    def CreatePageThree(self):
        label = Label( text="Quantization of Signals", font=25, bg='white',width=50,justify="center")
        label.place(x=400, y=20)
        label_file1 = Label( text="Input File:")
        label_file1.place(x=670, y=60)
        entreFile1 = Entry( width=50)
        entreFile1.place(x=550, y=90)

        button_file1 = Button( text="Browse",command=lambda :SelectFile1(entreFile1))
        button_file1.place(x=670, y=120)

        label_file2 = Label( text="Output File:")
        label_file2.place(x=660, y=160)

        entreFile2 = Entry( width=50)
        entreFile2.place(x=550, y=190)

        button_file2 = Button( text="Save As", command=lambda :SelectFile2(entreFile2))
        button_file2.place(x=670, y=220)

        label_choice = Label( text="Select number of bits or levels:")
        label_choice.place(x=610, y=250)

        combo_choice = ttk.Combobox( values=["Number of bits", "Number of levels"])
        combo_choice.place(x=625, y=280)
        combo_choice.set("Number of bits")  

        entry_choice = Entry( width=10)
        entry_choice.place(x=670,y=310)


        button_process = Button( text="Quantize signal", command= lambda :ProcessFilesForQuantization(entreFile1,entreFile2,combo_choice,entry_choice))
        button_process.place(x=660,y=340)

        test1 = Button( text="Quantization Test",width=25,height=2,background="white",command=lambda: QuantizationTest(combo_choice))
        test1.place(x=900, y=500)
    def CreatePageFour(self):
        label = Label( text=" Discrete Fourier Transform ", font=25, bg='white',width=50,justify="center")
        label.place(x=400, y=20)
        label_file1 = Label( text="Input File:")
        label_file1.place(x=670, y=60)
        entreFile1 = Entry( width=50)
        entreFile1.place(x=550, y=90)

        button_file1 = Button( text="Browse",command=lambda :SelectFile1(entreFile1))
        button_file1.place(x=670, y=120)

        label_file2 = Label( text="Output File:")
        label_file2.place(x=660, y=160)

        entreFile2 = Entry( width=50)
        entreFile2.place(x=550, y=190)

        button_file2 = Button( text="Save As", command=lambda :SelectFile2(entreFile2))
        button_file2.place(x=670, y=220)

        label_choice = Label( text="Select type of proccess:") 
        label_choice.place(x=630, y=250)

        combo_choice = ttk.Combobox( values=["DFT","IDFT"])
        combo_choice.place(x=625, y=280)
        combo_choice.set("DFT")  

  

        entry_choice = Entry( width=10)
        entry_choice.place(x=670,y=310)

        test1 = Button( text="Testing",width=25,height=2,background="white",command= CompareTask4)
        test1.place(x=900, y=500)
        def process_files():
         result = check_and_process(combo_choice, entreFile1, entreFile2, entry_choice)
         if isinstance(result, str):
          messagebox.showinfo("Result", result)

        button_process = Button(MainScreen, text="Process", command=process_files)
        button_process.pack()
        button_process.place(x=675,y=340)
    def CreatePageFive(self):
        label = Label(text="Discrete cosine Transform", font=25, bg='white', width=50, justify="center")
        label.place(x=400, y=20)

        labal2 = Label(text="Input File:")
        labal2.place(x=670, y=60)
        entreFile1 = Entry(width=50)
        entreFile1.place(x=550, y=90)

        buttonFile1 = Button(text="Load file", command=lambda: SelectFile1(entreFile1))
        buttonFile1.place(x=670, y=120)

        labal3 = Label(text="Output File:")
        labal3.place(x=660, y=160)

        entreFile2 = Entry(width=50)
        entreFile2.place(x=550, y=190)

        buttonFile2 = Button(text="Save As", command=lambda: SelectFile2(entreFile2))
        buttonFile2.place(x=670, y=220)
        proccessType = ttk.Combobox(values=["Sharpening", "DCT"])
        proccessType.place(x=625, y=280)
        proccessType.set("Sharpening")  # Default to Sharpening
        test = Button(text=" test", width=25, height=2, background="white")
        test.place(x=900, y=500)
        applyProccess = Button(text=" Apply process", width=11, background="white",command=lambda :sharpening(entreFile1, entreFile2))
        applyProccess.place(x=660,y=340)



    def CreatePageSix(self):
        label = Label(text="Task 6", font=25, bg='white', width=50, justify="center")
        label.place(x=400, y=20)

        labal2 = Label(text="Input File:")
        labal2.place(x=670, y=60)
        entreFile1 = Entry(width=50)
        entreFile1.place(x=550, y=90)

        buttonFile1 = Button(text="Load file", command=lambda: SelectFile1(entreFile1))
        buttonFile1.place(x=670, y=120)

        labal3 = Label(text="Output File:")
        labal3.place(x=660, y=160)

        entreFile2 = Entry(width=50)
        entreFile2.place(x=550, y=190)

        buttonFile2 = Button(text="Save As", command=lambda: SelectFile2(entreFile2))
        buttonFile2.place(x=670, y=220)

        labal4 = Label(text="Select type of process:")
        labal4.place(x=630, y=250)

        proccessType = ttk.Combobox(values=["Shifting", "Folding", "Shifting & Folding"])
        proccessType.place(x=625, y=280)
        proccessType.set("Shifting")  # Default to Shifting

        shiftingValue = Entry(width=10)
        shiftingValue.place(x=670, y=310)

        test = Button(text=" test", width=25, height=2, background="white")
        test.place(x=900, y=500)
        applyProccess = Button(text=" Apply process", width=11, background="white",
                    command=lambda: ChooseProccess(entreFile1, entreFile2, proccessType, shiftingValue))
        applyProccess.place(x=660,y=340)



    def next(self):
        if self.page < 6:  
            self.page += 1
            self.create_page()

    def back(self):
        if self.page > 0:
            self.page -= 1
            self.create_page()

if __name__ == "__main__":
    MainScreen = tk.Tk()
    app = App(MainScreen)
    MainScreen.geometry('1280x720')
    MainScreen.resizable(False, False)
    MainScreen.title('DSP Task')
    MainScreen.iconbitmap("DSP.ico")
    MainScreen.config(background='navajowhite')
    MainScreen.mainloop()



