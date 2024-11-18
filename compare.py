# Importing all Needed Libraries
from tkinter import *
from tkinter import messagebox
import math








# Task 2
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
# Task 3
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
# Task 4
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

# Task 5
def DCTSignalCompare(fileName,OutputInices,OutputSamples,m): 
    m = int(m.get())     
    ExpectedIndices=[]
    ExpectedSamples=[]
    with open(fileName, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                ExpectedIndices.append(V1)
                ExpectedSamples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(fileName)
    print("\n")
    if (m!=len(OutputSamples)) and (m!=len(OutputInices)):
        messagebox.showinfo("Error","DCT Test case failed, your signal have different length from the expected one")
        return
    for i in range(m):
        if(OutputInices[i]!=ExpectedIndices[i]):
            messagebox.showinfo("Error","DCT Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(m):
        if abs(OutputSamples[i] - ExpectedSamples[i]) < 0.01:
            continue
        else:
            messagebox.showinfo("Error","DCT Test case failed, your signal have different values from the expected one") 
            return
    messagebox.showinfo("Successful","DCT Test case passed successfully")
def SharpeningCompare(FirstDrev,SecondDrev):
    expectedOutput_first = [1, 1, 1, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    expectedOutput_second = [0, 0, 0, 0, 0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if( (len(FirstDrev)!=len(expectedOutput_first)) or (len(SecondDrev)!=len(expectedOutput_second))):
        # x =str(len(FirstDrev))
        # y = str(len(expectedOutput_first))
        # z = str (len(expectedOutput_second))
        # t = str ( len(SecondDrev))
        # print(x + "\n" + y + "\n" +t + "\n" + z + "\n")
        messagebox.showerror("Error","mismatch in length") 
        return
    first=second=True
    for i in range(len(expectedOutput_first)):
        if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
            continue
        else:
            first=False
            messagebox.showerror("Error","1st derivative wrong")
            return
    for i in range(len(expectedOutput_second)):
        if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
            continue
        else:
            second=False
            messagebox.showerror("Error","2nd derivative wrong") 
            return
    if(first and second):
        messagebox.showinfo("Successful","Derivative Test case passed successfully")
    else:
         messagebox.showerror("Error","Derivative Test case failed")
    return

# task 6
def ShiftFoldSignal(fileName,OutputInices,OutputSamples):      
    ExpectedIndices=[]
    ExpectedSamples=[]
    with open(fileName, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                ExpectedIndices.append(V1)
                ExpectedSamples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(fileName)
    print("\n")
    if (len(ExpectedSamples)!=len(OutputSamples)) and (len(ExpectedIndices)!=len(OutputInices)):
        messagebox.showinfo("Error","ShiftFoldSignal Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(OutputInices)):
        if(OutputInices[i]!=ExpectedIndices[i]):
            messagebox.showinfo("Error","ShiftFoldSignal Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(ExpectedSamples)):
        if abs(OutputSamples[i] - ExpectedSamples[i]) < 0.01:
            continue
        else:
            messagebox.showinfo("Error","ShiftFoldSignal Test case failed, your signal have different values from the expected one") 
            return
    messagebox.showinfo("Successful","ShiftFoldSignal Test case passed successfully")