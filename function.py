# Importing all Needed Libraries
from tkinter import *
from tkinter import ttk
from tkinter import Tk
from tkinter import filedialog
from tkinter import   filedialog, messagebox,END
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math







# Global variables
SignalType = 0
IsPeriodic = 0






# Task 1
def GenerateSignal(En1, En2, En3, En4, cmbo1):
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

            # Save wave data
            SaveWaveData(t, signal, t_discrete, discreteSignal, SamplingRate, SignalType, IsPeriodic)

            # Plot the continuous wave
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(t, signal, label='Continuous Wave', color='b')
            ax1.set_title('Continuous Wave')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True)
            ax1.legend()
            ax1.set_ylim([-1.5 * amplitude, 1.5 * amplitude])  # Set y-limits
            ax1.set_xlim([0, 0.01])  # Set x-limits
            plt.show()  # Show the continuous wave plot

            # Plot the discrete wave
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.stem(t_discrete, discreteSignal, linefmt='r-', markerfmt='ro', basefmt='k-', label='Discrete Wave')
            ax2.set_title('Discrete Wave')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True)
            ax2.legend()
            ax2.set_ylim([-1.5 * amplitude, 1.5 * amplitude])  # Set y-limits
            ax2.set_xlim([0, 0.01])  # Set x-limits
            plt.show()  # Show the discrete wave plot
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
            plt.stem(t, signal, basefmt=' ')  
            plt.title('Discrete-Time Sinusoidal Signal')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.xlim(0, duration)  
            plt.ylim(-1.5, 1.5)    
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
                file.write(f"{int(SignalType)}\n")
                file.write(f"{int(IsPeriodic)}\n")
                file.write(f"{int(SamplingRate)}\n")
                for index, amplitude in enumerate(signal):
                    file.write(f"{index} {amplitude:.5f}\n")
                file.write(f"{int(SignalType)}\n")
                file.write(f"{int(IsPeriodic)}\n")
                file.write(f"{int(SamplingRate)}\n")
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
                signal_type = int(file.readline().strip())  
                is_periodic = int(file.readline().strip())  
                N1 = int(file.readline().strip())           
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
def SignalSamplesAreEqual(file_name, samples):
    expected_indices = []
    expected_samples = []

    try:
        with open(file_name, 'r') as f:
            for _ in range(3):
                f.readline()

            for line in f:
                L = line.strip()
                if len(L.split()) == 2:
                    L = L.split()
                    V1 = int(L[0])
                    V2 = float(L[1])
                    expected_indices.append(V1)
                    expected_samples.append(V2)
                else:
                    break

        if len(expected_samples) != len(samples):
            messagebox.showerror("Error", "The length of the signal does not match the expected length")
            return

        for i in range(len(expected_samples)):
            if samples[i] != expected_samples[i]:  
                messagebox.showerror("Error", f"The signal value at index {i} does not match: expected {expected_samples[i]}, got {samples[i]}")
                return

        messagebox.showinfo("Success", "Test case passed successfully")

    except FileNotFoundError:
        messagebox.showerror("Error", "The file does not exist")

def CheckSamples():
    file1 = filedialog.askopenfilename(title="Select the first file")
    file2 = filedialog.askopenfilename(title="Select the second file")

    if file1 and file2:
        expected_samples = []
        with open(file1, 'r') as f:
            for _ in range(3):
                f.readline()
            for line in f:
                L = line.strip()
                if len(L.split()) == 2:
                    L = L.split()
                    V1 = int(L[0])
                    V2 = float(L[1])
                    expected_samples.append(V2)
                else:
                    break 

        SignalSamplesAreEqual(file2, expected_samples)  
    else:
        messagebox.showwarning("Warning", "You must select all files.")
# Task 2

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
            out_file.write(f" {int(sums[0])}\n")
            out_file.write(f" {int(sums[1])}\n")
            out_file.write(f" {int(sums[2])}\n")
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
def ChooseFileForAddition():
    file1 = filedialog.askopenfilename(title="Select the first file")
    file2 = filedialog.askopenfilename(title="Select the second file")
    output_file = filedialog.asksaveasfilename(title="Select output file", defaultextension=".txt")

    if file1 and file2 and output_file:
        AddFile(file1, file2, output_file)
    else:
        messagebox.showwarning("Warning", "You must select all files.")


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
            out_file.write(f" {int(sums)[0]}\n")
            out_file.write(f" {int(sums[1])}\n")
            out_file.write(f" {int(sums[2])}\n")
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
def ChooseFileForSubtraction():
    file1 = filedialog.askopenfilename(title="Select the first file")
    file2 = filedialog.askopenfilename(title="Select the second file")
    output_file = filedialog.asksaveasfilename(title="Select output file", defaultextension=".txt")

    if file1 and file2 and output_file:
        SubtractFile(file1, file2, output_file)
    else:
        messagebox.showwarning("Warning", "You must select all files.")     
 

def Multiplication(constEN):
    try:
        file1 = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        const = int(constEN.get())
        signal_dic ={}
        if not file1 :
           messagebox.showerror("Error","No file selected.")
           return
        
        with open(file1,'r') as file:
           lines = file.readlines()

        signal_type = int(lines[0].strip())
        isPeriodic = int(lines[1].strip())
        sampling_number = int(lines[2].strip())

        for line in lines[3:]:
            index , value = line.split()
            signal_dic[int(index)] = int(value)
        output_signal = {key: value * const for key , value in signal_dic.items()}
        output_file = filedialog.asksaveasfilename(filetypes=[("Text files", "*.txt")])
        if output_file:
            with open(output_file, 'w') as out_file:
                out_file.write(f"{signal_type} \n")
                out_file.write(f"{isPeriodic} \n")
                out_file.write(f"{sampling_number} \n")
                for key, value in output_signal.items():
                    out_file.write(f"{key} {value} \n")

            messagebox.showinfo("Success", "Output signal saved successfully.")
        else:
            messagebox.showwarning("Warning", "No output file selected.")
    except ValueError:
        messagebox.showerror("Error","Please provide a valid integer or file.")
    except Exception as e:
        messagebox.showerror("An error occurred", str(e))    

        
def Squaring():
    try:
        file1 = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        
        signal_dic ={}
        if not file1 :
           messagebox.showerror("Error","No file selected.")
           return
        
        with open(file1,'r') as file:
           lines = file.readlines()

        signal_type = int(lines[0].strip())
        isPeriodic = int(lines[1].strip())
        sampling_number = int(lines[2].strip())

        for line in lines[3:]:
            index , value = line.split()
            signal_dic[int(index)] = int(value)

        output_signal = {key: value * value for key , value in signal_dic.items()}

        output_file = filedialog.asksaveasfilename(filetypes=[("Text files", "*.txt")])
        
        if output_file:
            with open(output_file, 'w') as out_file:
                out_file.write(f"{signal_type} \n")
                out_file.write(f"{isPeriodic} \n")
                out_file.write(f"{sampling_number} \n")
                for key, value in output_signal.items():
                    out_file.write(f"{key} {value} \n")
            messagebox.showinfo("Success", "Output signal saved successfully.")
        else:
            messagebox.showwarning("Warning", "No output file selected.")
    except ValueError:
        messagebox.showerror("Error","Please provide a valid integer or file.")
    except Exception as e:
        messagebox.showerror("An error occurred", str(e))


def Normalization(file1,outputFile,cmbo2):
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
            out_file.write(f" {int(sums[0])}\n")
            out_file.write(f" {int(sums[1])}\n")
            out_file.write(f" {int(sums[2])}\n")
            all_indices = set(dict1.keys())
            if cmbo2.get() == "from 0 to 1":
                for index in sorted(all_indices):
                 value1 = dict1.get(index, 0)
                 result = (value1 - min_value)/(max_value-min_value)
                 result = round(result, 3)
                 out_file.write(f"{index} {result}\n")
            else:
                for index in sorted(all_indices):
                 value1 = dict1.get(index, 0)
                 result = (2* (value1 - min_value)/(max_value-min_value) )- 1
                 result = round(result, 3)
                 out_file.write(f"{index} {result}\n")

        messagebox.showinfo("Operation completed successfully", f"Results saved in {outputFile}")
    except FileNotFoundError:
        messagebox.showerror("Error", "One of the files is missing.")
    except Exception as e:
        messagebox.showerror("An error occurred", str(e))
def ChooseFileForNormalization(cmbo2):
    file1 = filedialog.askopenfilename(title="Select the first file")
    output_file = filedialog.asksaveasfilename(title="Select output file", defaultextension=".txt")

    if file1 and output_file:
        Normalization(file1, output_file,cmbo2)
    else:
        messagebox.showwarning("Warning", "You must select all files.")     


def Accumulation(file1,outputFile):
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
       
        result = 0
        with open(outputFile, 'w') as out_file:
            out_file.write(f" {int(sums[0])}\n")
            out_file.write(f" {int(sums[1])}\n")
            out_file.write(f" {int(sums[2])}\n")
            all_indices = set(dict1.keys())
            for index in sorted(all_indices):
                value1 = dict1.get(index, 0)
                result = result + value1
                out_file.write(f"{index} {result}\n")
        messagebox.showinfo("Operation completed successfully", f"Results saved in {outputFile}")
    except FileNotFoundError:
        messagebox.showerror("Error", "One of the files is missing.")
    except Exception as e:
        messagebox.showerror("An error occurred", str(e))
def ChooseFileForAccumulation():
    file1 = filedialog.askopenfilename(title="Select the first file")
    output_file = filedialog.asksaveasfilename(title="Select output file", defaultextension=".txt")

    if file1 and output_file:
        Accumulation(file1, output_file)
    else:
        messagebox.showwarning("Warning", "You must select all files.")     
# task 3
def QuantizeSignal(signal, numberOfLevels):
    minValue = np.min(signal)
    maxValue = np.max(signal)
    stepSize = (maxValue - minValue) / numberOfLevels
    midpoints = [minValue + (i + 0.5) * stepSize for i in range(numberOfLevels)]
    quantizedValues = []
    level_In_Binary = []
    levelIndices = []
    
    for sample in signal:
        levelIndex = min(int((sample - minValue) / stepSize), numberOfLevels - 1)
        quantized_value = midpoints[levelIndex]
        quantizedValues.append(quantized_value)
        binaryLevel = format(levelIndex, '0' + str(int(np.log2(numberOfLevels))) + 'b')
        level_In_Binary.append(binaryLevel)
        levelIndices.append(levelIndex + 1)

    return np.array(quantizedValues), midpoints, level_In_Binary, levelIndices

def SaveQuantizeData1(filename, skippedRows, quantizedValues, midpoints, level_In_Binary, levelIndices, originalSignal):
    with open(filename, 'w') as f:
        for row in skippedRows:
            f.write(f"{row}")
        
        # Calculate the difference between the original signal and quantized values
        differences =   quantizedValues - originalSignal 
        
        for quantized_value, binary, level_index, diff in zip(quantizedValues, level_In_Binary, levelIndices, differences):
            f.write(f"{level_index} {binary} {quantized_value:.3f} {diff:.3f}\n")

def SaveQuantizeData(filename, skippedRows, quantizedValues, level_In_Binary,  originalSignal):
    with open(filename, 'w') as f:
        for row in skippedRows:
            f.write(f"{row}")
        
        # Calculate the difference between the original signal and quantized values
        differences =  quantizedValues - originalSignal
        
        for binary, quantized_value, diff in zip(level_In_Binary, quantizedValues, differences):
            f.write(f"{binary} {quantized_value:.2f} \n")

def SelectFile1(entreFile1):
    filePath = filedialog.askopenfilename(title="Select Input File", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
    if filePath:
        entreFile1.delete(0, END)
        entreFile1.insert(0, filePath)

def SelectFile2(entreFile2):
    filePath = filedialog.asksaveasfilename(title="Select Output File", defaultextension=".txt", filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
    if filePath:
        entreFile2.delete(0, END)
        entreFile2.insert(0, filePath)

def ProcessFilesForQuantization(entreFile1,entreFile2,combo_choice,entry_choice):
    inputFile = entreFile1.get()
    outputFile = entreFile2.get()
    choice = combo_choice.get()
    choiceValue = entry_choice.get()

    if not inputFile or not outputFile or not choiceValue:
        messagebox.showwarning("Warning", "Please select both files and enter levels or bits.")
        return

    try:
        with open(inputFile, 'r') as f:
            lines = f.readlines()

        skippedRows = lines[:3]
        signal = np.array([float(line.split()[1].strip()) for line in lines[3:]])

        if choice == "Number of bits":
            numberOfBits = int(choiceValue)
            numberOfLevels = 2 ** numberOfBits
            quantizedValues, midpoints, level_In_Binary, levelIndices = QuantizeSignal(signal, numberOfLevels)
            SaveQuantizeData(outputFile, skippedRows, quantizedValues,  level_In_Binary, signal)
        else:
            numberOfLevels = int(choiceValue)
            quantizedValues, midpoints, level_In_Binary, levelIndices = QuantizeSignal(signal, numberOfLevels)
            SaveQuantizeData1(outputFile, skippedRows, quantizedValues, midpoints, level_In_Binary, levelIndices, signal)
        
        messagebox.showinfo("Success", f"Quantization complete!\nOutput saved to: {outputFile}")
    except Exception as e:
        messagebox.showerror("Error", str(e))
# testing 
def QuantizationTest1(file_name,Your_EncodedValues,Your_QuantizedValues):
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V2=int(L[0])
                V3=float(L[1])
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                line = f.readline()
            else:
                break
    if( (len(Your_EncodedValues)!=len(expectedEncodedValues)) or (len(Your_QuantizedValues)!=len(expectedQuantizedValues))):
        messagebox.showerror("Warning", "QuantizationTest1 Test case failed, your signal have different length from the expected one ")
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            messagebox.showerror("Warning", "QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one ")
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            messagebox.showerror("Warning", "QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one ")
            return
    messagebox.showinfo("Operation completed successfully", "QuantizationTest1 Test case passed successfully ")
def QuantizationTest2(file_name,Your_IntervalIndices,Your_EncodedValues,Your_QuantizedValues,Your_SampledError):
    expectedIntervalIndices=[]
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
    expectedSampledError=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==4:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                V3=float(L[2])
                V4=float(L[3])
                expectedIntervalIndices.append(V1)
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                expectedSampledError.append(V4)
                line = f.readline()
            else:
                break
    if(len(Your_IntervalIndices)!=len(expectedIntervalIndices)
     or len(Your_EncodedValues)!=len(expectedEncodedValues)
      or len(Your_QuantizedValues)!=len(expectedQuantizedValues)
      or len(Your_SampledError)!=len(expectedSampledError)):
        messagebox.showerror("Warning", "QuantizationTest2 Test case failed, your signal have different length from the expected one ")

        return
    for i in range(len(Your_IntervalIndices)):
        if(Your_IntervalIndices[i]!=expectedIntervalIndices[i]):
            messagebox.showerror("Warning", "QuantizationTest2 Test case failed, your signal have different indicies from the expected one")

            return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            messagebox.showerror("Warning", "QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one")

            return
        
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            messagebox.showerror("Warning", "QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one ")

            return
    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
            continue
        else:
            messagebox.showerror("Warning", "QuantizationTest2 Test case failed, your SampledError have different values from the expected one ")


            return
    messagebox.showinfo("Operation completed successfully", "QuantizationTest2 Test case passed successfully")

def QuantizationTest(cmbo):
    file1 = filedialog.askopenfilename(title="Select the first file")
    file2 = filedialog.askopenfilename(title="Select the second file")
    if file1 and file2:
        if cmbo.get() == "Number of bits":
            expected_samples = []
            expected_index = []
            with open(file1, 'r') as f:
                for _ in range(3):
                    f.readline()
                for line in f:
                    L = line.strip()
                    if len(L.split()) == 2:
                        L = L.split()
                        V1 = int(L[0])
                        V2 = float(L[1])
                        expected_samples.append(V2)
                        expected_index.append(V1)
                    else:
                        break 

            QuantizationTest1(file2,expected_index,expected_samples)
        else :   
            expected_samples = []
            expected_index = []
            IntervalIndices=[]
            SampledError=[]
            with open(file1, 'r') as f:
                for _ in range(3):
                    f.readline()
                for line in f:
                    L = line.strip()
                    if len(L.split()) == 4:
                        L = L.split()
                        V1 = int(L[0])
                        V2 = float(L[1])
                        V3 = float(L[2])
                        V4 = float(L[3])
                        expected_index.append(V2)
                        expected_samples.append(V3)
                        IntervalIndices.append(V1)
                        SampledError.append(V4)
                    else:
                        break 

            QuantizationTest2(file2,IntervalIndices,expected_index,expected_samples,SampledError)

    else:
        messagebox.showwarning("Warning", "You must select all files.")
        
# Task 4
def ReadFrequencyComponents(file_path):
    frequencyComponents = []
    skippedRows = []
    try:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i < 3: 
                    skippedRows.append(line.strip())  # Save the first 3 rows
                    continue
                line = line.strip().replace('f', '')  # Remove unwanted 'f' characters
                try:
                    magnitude, phase = map(float, line.split(','))  # Convert to float
                    frequencyComponents.append((magnitude, phase))
                except ValueError:
                    print(f"Skipping malformed line {i + 1}: {line}")
        return skippedRows, frequencyComponents
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None, None

# Function to convert polar to rectangular form
def Convert(frequencyComponents):
    convertComponents = []
    for magnitude, phase in frequencyComponents:
        real = magnitude * np.cos(phase)  
        imag = magnitude * np.sin(phase)  
        convertComponents.append(real + 1j * imag) 
    return np.array(convertComponents)

# Function to apply IDFT
def IDFTConvert(convertComponents):
    N = len(convertComponents)
    timeSignal = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            timeSignal[n] += convertComponents[k] * np.exp(1j * (2 * np.pi / N) * k * n)
    return timeSignal / N  # Normalize

# Function to process the files for DFT
def ProcessFilesForiDFT(entry_file1, entry_file2):
    input_file = entry_file1.get()
    output_file = entry_file2.get()

    if not input_file or not output_file:
        return "Please select both input and output files."
    
    # Read frequency components
    skippedRows, frequencyComponents = ReadFrequencyComponents(input_file)
    
    if frequencyComponents is None:
        return "Error reading frequency components."

    # Convert polar to rectangular components
    convertComponents = Convert(frequencyComponents)

    # Apply IDFT to get the time-domain signal
    reconstructed_signal = IDFTConvert(convertComponents)

    # Write to output file
    with open(output_file, 'w') as out_file:
        # Write the first three rows (header)
        for line in skippedRows:
            out_file.write(line + '\n')

        # Write the index, magnitude
        for index, value in enumerate(reconstructed_signal):
            realPart = round(value.real, 2)
            imagPart = round(value.imag, 2)
            magnitude = int(round(np.sqrt(realPart**2 + imagPart**2), 2))  # Calculate magnitude
            out_file.write(f"{index} {magnitude}\n")

    return "Processing complete. Output saved."

# Function to compute DFT


def DFT(file_path, output_path, sampling_frequency):
    if not file_path:
        return "Please select a file."
    
    if not output_path:
        return "Please specify an output file."

    try:
        # Read input data (ignoring first three rows)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data = []
            signal_type = int(lines[0].strip())
            is_periodic = int(lines[1].strip())
            N1 = int(lines[2].strip())
            for line in lines[3:]:
                parts = line.strip().split()
                if len(parts) == 2:
                    time, amplitude = map(float, parts)
                    data.append(amplitude)

        signal_data = np.array(data)
        N = len(signal_data)

        frequency_components = []
        for k in range(N):
            real_sum = 0
            imag_sum = 0
            for n in range(N):
                angle = -2 * np.pi * k * n / N
                real_sum += signal_data[n] * np.cos(angle)
                imag_sum += signal_data[n] * np.sin(angle)
            frequency_components.append(complex(real_sum, imag_sum))

        # All N frequencies
        frequencies = np.arange(N) * (sampling_frequency / N)
        amplitudes = np.abs(frequency_components)
        phases = np.angle(frequency_components)
        theta = ((2*3.14)/(N*(1/sampling_frequency)))
        
        # Plotting frequency vs amplitude and frequency vs phase
        plt.figure(figsize=(8, 6))




        
        result_list = [theta * i for i in range(1, N+1)]
        print(result_list)

        plt.subplot(2, 1, 1)
        plt.stem(result_list, amplitudes, basefmt=" ")
        plt.title('Frequency vs amplitudes ')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Adjusted Phase (radians)')
        plt.grid()   
        # Frequency vs Phase
        plt.subplot(2, 1, 2)
        plt.stem(result_list, phases, basefmt=" ")
        plt.title('Frequency vs Adjusted Phase ')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Adjusted Phase (radians)')
        plt.grid()   

        plt.tight_layout()
        plt.show()

        # Save output in the specified format
        with open(output_path, 'w') as outfile:
            outfile.write(f"{signal_type} \n")
            is_periodic = 1
            outfile.write(f"{is_periodic} \n")
            outfile.write(f"{N1} \n")

            for i in range(N):
                amplitude_str = f"{int(amplitudes[i])}" if amplitudes[i].is_integer() else f"{amplitudes[i]:.14f}f"
                phase_str = f"{int(phases[i])}" if phases[i].is_integer() else f"{phases[i]:.14f}f"
                outfile.write(f"{amplitude_str} {phase_str}\n")

        return f"DFT results saved to {output_path}."

    except ValueError:
        return "Invalid data or sampling frequency."
    except Exception as e:
        return f"An error occurred: {e}"

    


# Function to check combo box selection and call appropriate function
def check_and_process(combo_box, entry_file, entry_output_file, entry_sampling_freq):
    selected_option = combo_box.get()
    if selected_option == "DFT":
        output_path = entry_output_file.get()
        file_path = entry_file.get()
        sampling_frequency = entry_sampling_freq.get()
        if not sampling_frequency:
            return "Please enter a sampling frequency."
        try:
            sampling_frequency = float(sampling_frequency)
            result = DFT(file_path, output_path, sampling_frequency)
            return result
        except ValueError:
            return "Invalid sampling frequency."
    else:
        result = ProcessFilesForiDFT(entry_file, entry_output_file)
        return result
def SignalCompareAmplitude(SignalInput=[], SignalOutput=[]):
    if len(SignalInput) != len(SignalOutput):
        return False
    for i in range(len(SignalInput)):
        if abs(SignalInput[i] - SignalOutput[i]) > 0.001:
            return False
    return True

def RoundPhaseShift(P):
    while P < 0:
        P += 2 * math.pi
    return float(P % (2 * math.pi))

def SignalComparePhaseShift(SignalInput=[], SignalOutput=[]):
    if len(SignalInput) != len(SignalOutput):
        return False
    for i in range(len(SignalInput)):
        A = round(SignalInput[i], 4)  # Round to 4 decimal places for comparison
        B = round(SignalOutput[i], 4)
        if abs(A - B) > 0.0001:
            return False
    return True 

def CompareTask3():
    file1 = filedialog.askopenfilename(title="Select the first file")
    file2 = filedialog.askopenfilename(title="Select the second file")
    
    if file1 and file2:
        samplesfile1 = []
        indexfile1 = []
        samplesfile2 = []
        indexfile2 = []

        # Read first file
        with open(file1, 'r') as f:
            for _ in range(3):
                f.readline()  # Skip first three lines
            for line in f:
                L = line.strip().replace('f', '')  # Remove 'f' before processing
                if len(L.split()) == 2:
                    V1, V2 = map(float, L.split())
                    indexfile1.append(V1)
                    samplesfile1.append(V2)

        # Read second file
        with open(file2, 'r') as f:
            for _ in range(3):
                f.readline()  # Skip first three lines
            for line in f:
                L = line.strip().replace('f', '')  # Remove 'f' before processing
                if len(L.split()) == 2:
                    V1, V2 = map(float, L.split())
                    indexfile2.append(V1)
                    samplesfile2.append(V2)

        # Store the results of the comparison functions
        amplitude_result = SignalCompareAmplitude(indexfile1, indexfile2)
        phase_result = SignalComparePhaseShift(samplesfile1, samplesfile2)

        # Check the results
        if amplitude_result and phase_result:
            messagebox.showinfo("Result", "Test case passed successfully")
        else:
            messagebox.showwarning("Warning", "Invalid values.")
