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

# def generate():
#     if En2.get <= 100:
#         GenerateSignal2()
#     else:
#         GenerateSignal()            
def GenerateSignal():

    # Get parameters from input fields
    try:
        amplitude = float(En1.get())         # Amplitude
        frequency = float(En2.get())         # Frequency in Hz
        PhaseShift = float(En4.get())        # Phase shift in radians
        SamplingRate = float(En3.get())      # Sampling rate in Hz
        if frequency >= 100 :
              # Generate time values
            t = np.linspace(0, 1, int(SamplingRate ), endpoint=False)
            if cmbo1.get() == "Sin":
                signal = amplitude * np.sin(2 * np.pi * frequency * t + PhaseShift)
            else:   
                signal = amplitude * np.cos(2 * np.pi * frequency * t + PhaseShift)
            # Discrete wave sampling
            n_samples = int(frequency)
            t_discrete = np.linspace(0, 1, n_samples,endpoint=False)
            if  cmbo1.get() == 'Sin':
                discreteSignal = amplitude * np.sin(2 * np.pi * frequency * t_discrete + PhaseShift)
            elif cmbo1.get() == 'Cos':
                discreteSignal = amplitude * np.cos(2 * np.pi * frequency * t_discrete + PhaseShift)
            SaveWaveData(t, signal, t_discrete, discreteSignal)
            for widget in plot.winfo_children():
                    widget.destroy()
            # Create a new figure with a larger size
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))

            # Plot the continuous wave
            ax1.plot(t, signal, label='Continuous Wave', color='b')
            ax1.set_title('Continuous Wave')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.legend()
            ax1.grid(True)

            # Plot the discrete wave
            ax2.stem(t_discrete, discreteSignal, linefmt='r-', markerfmt='ro', basefmt='k-', label='Discrete Wave')
            ax2.set_title('Discrete Wave')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.legend()
            ax2.grid(True)

            # Set larger axis limits for better visualization
            ax1.set_ylim([-1.5 * amplitude, 1.5* amplitude])
            ax2.set_ylim([-1.5 * amplitude, 1.5 * amplitude])

            ax1.set_xlim([0, 0.01])  # Set limits for the continuous wave
            ax2.set_xlim([0, 0.01])  # Set limits for the discrete wave

            # Display the figure in the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=plot)
            canvas.draw()
            canvas.get_tk_widget().pack()
        else:
            if cmbo1.get() == "Sin":
            # Step 1: Define parameters
                amplitude = float(En1.get())         # Amplitude of the sine wave
                frequency = float(En2.get())         # Frequency in Hz
                PhaseShift = float(En4.get())       # Phase shift in radians
                duration = 2          # Duration in seconds
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
                duration = 2          # Duration in seconds
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

      

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all parameters.")
        return
def SaveWaveData(t, signal, t_discrete, discreteSignal):
    """Saves the generated wave data to a file."""
    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text files", ".txt"), ("All files", ".*")],
                                             title="Save Wave Data")
    if file_path:
        try:
            with open(file_path, 'w') as file:
                # Save continuous wave data
                file.write("Continuous Wave Data:\n")
                file.write("Index, Amplitude\n")
                for index, amplitude in enumerate(signal):
                    file.write(f"{index}, {amplitude:.5f}\n")

                file.write("\nDiscrete Wave Data:\n")
                file.write("Index, Amplitude\n")
                for index, amplitude in enumerate(discreteSignal):
                    file.write(f"{index}, {amplitude:.5f}\n")

            messagebox.showinfo("Success", "Wave data saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save wave data: {e}")

def read_file():
    filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    
    if filepath:
        try:
            with open(filepath, "r") as file:
                # Read metadata
                signal_type = int(file.readline().strip())  # 0 for time, 1 for frequency
                is_periodic = int(file.readline().strip())  # 0 or 1 for periodicity
                N1 = int(file.readline().strip())           # Number of samples or frequencies

                # Read data based on signal type
                if signal_type == 0:
                    # Time domain data (Sample Index, Amplitude)
                    data = np.loadtxt(file, max_rows=N1)
                    time = data[:, 0]
                    amplitude = data[:, 1]

                    # Plot continuous time-domain signal
                    plt.figure(figsize=(10, 5))
                    plt.plot(time, amplitude, label='Time Domain Signal', color='blue')
                    plt.title('(Continuous)')
                    plt.xlabel('time (s)')
                    plt.ylabel('Amplitude')
                    plt.grid(True)
                    plt.legend()
                    plt.show()

                    # Plot discrete time-domain signal
                    plt.figure(figsize=(10, 5))
                    plt.stem(time, amplitude, basefmt=' ')
                    plt.title('(Discrete)')
                    plt.xlabel('Sample Index')
                    plt.ylabel('Amplitude')
                    plt.grid(True)
                    plt.show()

                elif signal_type == 1:
                    # Frequency domain data (Frequency, Amplitude, Phase Shift)
                    data = np.loadtxt(file, max_rows=N1)
                    frequency = data[:, 0]
                    amplitude = data[:, 1]
                    phase_shift = data[:, 2]

                    # Plot frequency domain signal (Amplitude vs Frequency)
                    plt.figure(figsize=(10, 5))
                    plt.plot(frequency, amplitude, label='Frequency Domain Signal', color='blue')
                    plt.title('(Continuous)')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Amplitude')
                    plt.grid(True)
                    plt.legend()
                    plt.show()

                    # Plot phase shift
                    plt.figure(figsize=(10, 5))
                    plt.plot(frequency, phase_shift, label='Phase Shift', color='green')
                    plt.title('Frequency Domain Phase Shift')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('Phase Shift (radians)')
                    plt.grid(True)
                    plt.legend()
                    plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {e}")
# def GenerateSignal2():






# Creating the main window
MainScreen = Tk()
MainScreen.geometry('1700x1080')
MainScreen.resizable(False , False)
MainScreen.title('DSP Task')
# MainScreen.iconbitmap("AI.ico")
MainScreen.config(background='silver')
fr1  = Frame(width='1900',height='1060',bg='white')
fr2  = Frame(fr1,width='350',height='280',bg='silver')
fr3  = Frame(fr1,width='300',height='280',bg='silver')
plot = Frame(width='1900',height='400',bg='white')

Lb1  = Label(text=' Wave type ',fg='black',bg='silver',font=25,width=10)
Lb2  = Label(fr2,text=' Amplitude ',fg='black',bg='white',font=25,width=25)
Lb3  = Label(fr2,text=' AnalogFrequency ',fg='black',bg='white',font=25,width=25)
Lb4  = Label(fr2,text=' SamplingFrequency',fg='black',bg='white',font=25,width=25)
Lb5  = Label(fr2,text=' PhaseShift ',fg='black',bg='white',font=25,width=25)
Lb9  = Label(fr1,text=' Welcome TO Signal Digital Program ',fg='black',bg='silver',font=25,width=35)
Lb7  = Label(fr3,text=' Inputs ',fg='black',bg='white',font=25,width=20)
Lb8  = Label(fr2,text=' Signal Information ',fg='black',bg='white',font=25,width=25)



bt1  = Button(MainScreen,text='Generate',fg='black',bg='silver',width=30,height=2,command=GenerateSignal)  
bt2  = Button(MainScreen,text='Open Folder',fg='black',bg='silver',command=read_file,width=30,height=2)

En1 = Entry(fr3,fg='black',bg='white',font= 15,justify="center")
En2 = Entry(fr3,fg='black',bg='white',font= 15,justify="center")
En3 = Entry(fr3,fg='black',bg='white',font= 15,justify="center")
En4 = Entry(fr3,fg='black',bg='white',font= 15,justify="center")



bt1.place(x=680,y=400)
bt2.place(x=930,y=400)

fr1.place(x=5,y=5)
fr2.place(x=30,y=80)
fr3.place(x=400,y=80 )
plot.place(x=5,y=420)

En1.place(x=40,y=60)
En2.place(x=40,y=100)
En3.place(x=40,y=140)
En4.place(x=40,y=180)


Lb1.place(x=150,y=400)
Lb2.place(x=40,y=60)
Lb3.place(x=40,y=100)
Lb4.place(x=40,y=140)
Lb5.place(x=40,y=180)
Lb7.place(x=40,y=20)
Lb8.place(x=40,y=20)
Lb9.place(x=400,y=30)


cmbo1 = ttk.Combobox(MainScreen, value = ('Sin','Cos'),background ='silver')
cmbo1.set('Sin')
cmbo1.place(x=340,y=400,height=30,width=50)
MainScreen.mainloop()



