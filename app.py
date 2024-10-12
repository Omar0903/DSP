# Importing all Needed Libraries
from tkinter import *
from tkinter import ttk
from tkinter import Tk
from tkinter import filedialog
from tkinter import   filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def GenerateSignal():
    if cmbo1.get() == "Sin":
            # Step 1: Define parameters
        amplitude = float(En1.get())         # Amplitude of the sine wave
        frequency = float(En2.get())         # Frequency in Hz
        PhaseShift = float(En4.get())       # Phase shift in radians
        duration = float(En5.get())          # Duration in seconds
        SamplingRate =  float(En3.get())  # Sampling rate in Hz
        # continuous Sinusoidal Signals 
        # Step 2: Generate time values
        t = np.linspace(0, duration, int(SamplingRate * duration), endpoint=False)
        # Step 3: Compute the sine wave
        signal = amplitude * np.sin(2 * np.pi * frequency * t + PhaseShift)
        # Step 4: Plot the sine wave
        plt.figure(figsize=(10, 5))
        plt.plot(t, signal, label='Sine Wave', color='blue')
        plt.title('Sine Wave')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show()
        # Discrete Sinusoidal Signals
        signal = amplitude * np.sin(2 * np.pi * frequency * t + PhaseShift)
        plt.figure(figsize=(10, 5))
        plt.stem(t, signal, basefmt=' ')  # Remove use_line_collection
        plt.title('Discrete-Time Sinusoidal Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.xlim(0, duration)  # Limit x-axis to the duration
        plt.ylim(-1.5, 1.5)    # Set y-axis limits
        plt.show()
        with open("sine_wave.txt", "w") as file:
            for index, value in enumerate(signal):
                file.write(f"{index}: {value}\n")

        print(" The data has been stored. 'sine_wave.txt'")
    else:
           # Step 1: Define parameters
        amplitude = float(En1.get())         # Amplitude of the sine wave
        frequency = float(En2.get())         # Frequency in Hz
        PhaseShift = float(En4.get())       # Phase shift in radians
        duration = float(En5.get())          # Duration in seconds
        SamplingRate =  float(En3.get())  # Sampling rate in Hz
        # continuous Sinusoidal Signals 
        # Step 2: Generate time values
        t = np.linspace(0, duration, int(SamplingRate * duration), endpoint=False)
        # Step 3: Compute the sine wave
        signal = amplitude * np.cos(2 * np.pi * frequency * t + PhaseShift)
        # Step 4: Plot the sine wave
        plt.figure(figsize=(10, 5))
        plt.plot(t, signal, label='Sine Wave', color='blue')
        plt.title('Cosine Wave')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show()
        # Discrete Sinusoidal Signals
        signal = amplitude * np.cos(2 * np.pi * frequency * t + PhaseShift)
        plt.figure(figsize=(10, 5))
        plt.stem(t, signal, basefmt=' ')  # Remove use_line_collection
        plt.title('Discrete-Time Sinusoidal Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.xlim(0, duration)  # Limit x-axis to the duration
        plt.ylim(-1.5, 1.5)    # Set y-axis limits
        plt.show()
        with open("cosine_wave.txt", "w") as file:
            for value in signal:
                file.write(f"{value}\n")

        print(" The data has been stored 'cosine.txt'")



def read_file():
    filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    
    if filepath:
        try:
            with open(filepath, "r") as file:
                content = file.read()
                messagebox.showinfo("File Content", content)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {e}")
# Creating the main window
MainScreen = Tk()
MainScreen.geometry('1200x720')
MainScreen.resizable(False , False)
MainScreen.title('DSP Task')
# MainScreen.iconbitmap("AI.ico")
MainScreen.config(background='silver')
fr1  = Frame(width='1150',height='680',bg='white')
fr2  = Frame(fr1,width='350',height='280',bg='silver')
fr3  = Frame(fr1,width='300',height='280',bg='silver')

Lb1  = Label(text=' Wave type ',fg='black',bg='silver',font=25,width=10)
Lb2  = Label(fr2,text=' Amplitude ',fg='black',bg='white',font=25,width=25)
Lb3  = Label(fr2,text=' AnalogFrequency ',fg='black',bg='white',font=25,width=25)
Lb4  = Label(fr2,text=' SamplingFrequency',fg='black',bg='white',font=25,width=25)
Lb5  = Label(fr2,text=' PhaseShift ',fg='black',bg='white',font=25,width=25)
Lb6  = Label(fr2,text=' duration ',fg='black',bg='white',font=25,width=25)
Lb9  = Label(fr1,text=' Welcome TO Signal Digital Program ',fg='black',bg='silver',font=25,width=35)
Lb7  = Label(fr3,text=' Inputs ',fg='black',bg='white',font=25,width=20)
Lb8  = Label(fr2,text=' Signal Information ',fg='black',bg='white',font=25,width=25)



bt1  = Button(MainScreen,text='Generate',fg='black',bg='silver',width=30,height=2,command=GenerateSignal)  
bt2  = Button(MainScreen,text='Open Folder',fg='black',bg='silver',command=read_file,width=30,height=2)

En1 = Entry(fr3,fg='black',bg='white',font= 15,justify="center")
En2 = Entry(fr3,fg='black',bg='white',font= 15,justify="center")
En3 = Entry(fr3,fg='black',bg='white',font= 15,justify="center")
En4 = Entry(fr3,fg='black',bg='white',font= 15,justify="center")
En5 = Entry(fr3,fg='black',bg='white',font= 15,justify="center")


bt1.place(x=680,y=400)
bt2.place(x=930,y=400)

fr1.place(x=25,y=20)
fr2.place(x=30,y=80)
fr3.place(x=400,y=80 )

En1.place(x=40,y=60)
En2.place(x=40,y=100)
En3.place(x=40,y=140)
En4.place(x=40,y=180)
En5.place(x=40,y=220)

Lb1.place(x=150,y=400)
Lb2.place(x=40,y=60)
Lb3.place(x=40,y=100)
Lb4.place(x=40,y=140)
Lb5.place(x=40,y=180)
Lb6.place(x=40,y=220)
Lb7.place(x=40,y=20)
Lb8.place(x=40,y=20)
Lb9.place(x=400,y=30)


cmbo1 = ttk.Combobox(MainScreen, value = ('Sin','Cos'),background ='silver')
cmbo1.set('Sin')
cmbo1.place(x=340,y=400,height=30,width=50)
MainScreen.mainloop()



