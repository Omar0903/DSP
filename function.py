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
SignalType  = 0
IsPeriodic = 0

def new_1(file1,outputFile):
    try:
        dict1 = {}
        sums = [0, 0, 0]  
        with open(file1, 'r') as f1:
            for i in range(3):
                line = f1.readline()
                if line:
                    value = float(line.strip())
                    sums[i] += value 
            for line in f1:
                index, value = line.split()
                dict1[int(index)] = float(value)
        max_value = max(dict1.values())
        min_value = min(dict1.values())
        with open(outputFile, 'w') as out_file:
            out_file.write(f" {sums[0]}\n")
            out_file.write(f" {sums[1]}\n")
            out_file.write(f" {sums[2]}\n")
            all_indices = set(dict1.keys())
            for index in sorted(all_indices):
                value1 = dict1.get(index, 0)
                result = (value1 - min_value)/(max_value-min_value)
                out_file.write(f"{index} {result}\n")
        messagebox.showinfo("Operation completed successfully", f"Results saved in {outputFile}")
    except FileNotFoundError:
        messagebox.showerror("Error", "One of the files is missing.")
    except Exception as e:
        messagebox.showerror("An error occurred", str(e))


def AddFile(file1, file2, outputFile):
    try:
        dict1 = {}
        dict2 = {}
        sums = [0, 0, 0]  
        with open(file1, 'r') as f1:
            for i in range(3):
                line = f1.readline()
                if line:
                    value = float(line.strip())
                    sums[i] += value / 2  
            for line in f1:
                index, value = line.split()
                dict1[int(index)] = float(value)
        with open(file2, 'r') as f2:
            for i in range(3):
                line = f2.readline()
                if line:
                    value = float(line.strip())
                    sums[i] += value / 2  
            for line in f2:
                index, value = line.split()
                dict2[int(index)] = float(value)
        with open(outputFile, 'w') as out_file:
            out_file.write(f" {sums[0]}\n")
            out_file.write(f" {sums[1]}\n")
            out_file.write(f" {sums[2]}\n")
            all_indices = set(dict1.keys()).union(set(dict2.keys()))
            for index in sorted(all_indices):
                value1 = dict1.get(index, 0)
                value2 = dict2.get(index, 0)
                result = value1 + value2
                out_file.write(f"{index} {result}\n")
        messagebox.showinfo("Operation completed successfully", f"Results saved in {outputFile}")
    except FileNotFoundError:
        messagebox.showerror("Error", "One of the files is missing.")
    except Exception as e:
        messagebox.showerror("An error occurred", str(e))
def SubtractFile(file1, file2, outputFile):
    try:
        dict1 = {}
        dict2 = {}
        sums = [0, 0, 0]  
        with open(file1, 'r') as f1:
            for i in range(3):
                line = f1.readline()
                if line:
                    value = float(line.strip())
                    sums[i] += value / 2  
            for line in f1:
                index, value = line.split()
                dict1[int(index)] = float(value)
        with open(file2, 'r') as f2:
            for i in range(3):
                line = f2.readline()
                if line:
                    value = float(line.strip())
                    sums[i] += value / 2  
            for line in f2:
                index, value = line.split()
                dict2[int(index)] = float(value)
        with open(outputFile, 'w') as out_file:
            out_file.write(f" {sums[0]}\n")
            out_file.write(f" {sums[1]}\n")
            out_file.write(f" {sums[2]}\n")
            all_indices = set(dict1.keys()).union(set(dict2.keys()))
            for index in sorted(all_indices):
                value1 = dict1.get(index, 0)
                value2 = dict2.get(index, 0)
                result = value1 - value2
                out_file.write(f"{index} {result}\n")
        messagebox.showinfo("Operation completed successfully", f"Results saved in {outputFile}")
    except FileNotFoundError:
        messagebox.showerror("Error", "One of the files is missing.")
    except Exception as e:
        messagebox.showerror("An error occurred", str(e))        
def ChooseFileForAddition():
    file1 = filedialog.askopenfilename(title="Select the first file")
    file2 = filedialog.askopenfilename(title="Select the second file")
    output_file = filedialog.asksaveasfilename(title="Select output file", defaultextension=".txt")

    if file1 and file2 and output_file:
        AddFile(file1, file2, output_file)
    else:
        messagebox.showwarning("Warning", "You must select all files.")
def ChooseFileForSubtraction():
    file1 = filedialog.askopenfilename(title="Select the first file")
    file2 = filedialog.askopenfilename(title="Select the second file")
    output_file = filedialog.asksaveasfilename(title="Select output file", defaultextension=".txt")

    if file1 and file2 and output_file:
        SubtractFile(file1, file2, output_file)
    else:
        messagebox.showwarning("Warning", "You must select all files.")     
def ChooseFileForNormalization():
    file1 = filedialog.askopenfilename(title="Select the first file")
    output_file = filedialog.asksaveasfilename(title="Select output file", defaultextension=".txt")

    if file1 and output_file:
        new_1(file1, output_file)
    else:
        messagebox.showwarning("Warning", "You must select all files.")     
def GenerateSignal(En1, En2, En3, En4, cmbo1, plot):
    try:
        amplitude = float(En1.get())        
        frequency = float(En2.get())         
        PhaseShift = float(En4.get())        
        SamplingRate = int(En3.get())     
        duration = 2  
        if frequency >= 100:
            # Generate time values
            t = np.linspace(0, 1, int(SamplingRate), endpoint=False)
            if cmbo1.get() == "Sin":
                signal = amplitude * np.sin(2 * np.pi * frequency * t + PhaseShift)
            else:   
                signal = amplitude * np.cos(2 * np.pi * frequency * t + PhaseShift)
            # Discrete wave sampling
            n_samples = int(frequency)
            t_discrete = np.linspace(0, 1, n_samples, endpoint=False)
            if cmbo1.get() == 'Sin':
                discreteSignal = amplitude * np.sin(2 * np.pi * frequency * t_discrete + PhaseShift)
            elif cmbo1.get() == 'Cos':
                discreteSignal = amplitude * np.cos(2 * np.pi * frequency * t_discrete + PhaseShift)
            SaveWaveData(t, signal, t_discrete, discreteSignal, SamplingRate,SignalType,IsPeriodic)
            for widget in plot.winfo_children():
                widget.destroy()
            # Create a new figure with a larger size
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
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
            ax1.set_ylim([-1.5 * amplitude, 1.5 * amplitude])
            ax2.set_ylim([-1.5 * amplitude, 1.5 * amplitude])
            ax1.set_xlim([0, 0.01]) 
            ax2.set_xlim([0, 0.01])
            # Display the plots on Tkinter window
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            canvas = FigureCanvasTkAgg(fig, master=plot)
            canvas.draw()
            canvas.get_tk_widget().pack()
        else:
            # Code for the case where frequency is less than 100
            t = np.linspace(0, duration, int(SamplingRate * duration), endpoint=False)
            if cmbo1.get() == "Sin":
                 signal = amplitude * np.sin(2 * np.pi * frequency * t + PhaseShift)
            else:   
                 signal = amplitude * np.cos(2 * np.pi * frequency * t + PhaseShift)        
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

            # Save the generated sine wave data to a text file
            with open("sine_wave.txt", "w") as file:
                for index, value in enumerate(signal):
                    file.write(f"{index} {value}\n")
            messagebox.showinfo("Success", "Wave data saved successfully!")    
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all parameters.")
        return

def SaveWaveData(t, signal, t_discrete, discreteSignal, SamplingRate,IsPeriodic,SignalType):
    """Saves the generated wave data to a file."""
    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text files", ".txt"), ("All files", ".*")],
                                             title="Save Wave Data")
    if file_path:
        try:
            with open(file_path, 'w') as file:
                # Save continuous wave data
                file.write(f"{SignalType}\n")
                file.write(f"{IsPeriodic}\n")
                file.write(f"{SamplingRate}\n")
                for index, amplitude in enumerate(signal):
                    file.write(f"{index} {amplitude:.5f}\n")
                file.write(f"{SignalType}\n")
                file.write(f"{IsPeriodic}\n")
                file.write(f"{SamplingRate}\n")
                for index, amplitude in enumerate(discreteSignal):
                    file.write(f"{index}, {amplitude:.5f}\n")

            messagebox.showinfo("Success", "Wave data saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save wave data: {e}")
def ReadFile():
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