# Importing all Needed Libraries
from tkinter import *
from tkinter import filedialog
from tkinter import filedialog, messagebox, END
import numpy as np
import matplotlib.pyplot as plt
from compare import *


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
            if cmbo1.get() == "Sin":
                discreteSignal = amplitude * np.sin(
                    2 * np.pi * frequency * t_discrete + PhaseShift
                )
            elif cmbo1.get() == "Cos":
                discreteSignal = amplitude * np.cos(
                    2 * np.pi * frequency * t_discrete + PhaseShift
                )

            # Save wave data
            SaveWaveData(
                t,
                signal,
                t_discrete,
                discreteSignal,
                SamplingRate,
                SignalType,
                IsPeriodic,
            )

            # Plot the continuous wave
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(t, signal, label="Continuous Wave", color="b")
            ax1.set_title("Continuous Wave")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True)
            ax1.legend()
            ax1.set_ylim([-1.5 * amplitude, 1.5 * amplitude])  # Set y-limits
            ax1.set_xlim([0, 0.01])  # Set x-limits
            plt.show()  # Show the continuous wave plot

            # Plot the discrete wave
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.stem(
                t_discrete,
                discreteSignal,
                linefmt="r-",
                markerfmt="ro",
                basefmt="k-",
                label="Discrete Wave",
            )
            ax2.set_title("Discrete Wave")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Amplitude")
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
            plt.plot(t, signal, label="Sine Wave", color="blue")
            plt.title("Sine Wave")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()
            plt.show()

            # Discrete Sinusoidal Signals
            signal = amplitude * np.sin(2 * np.pi * frequency * t + PhaseShift)
            plt.figure(figsize=(10, 5))
            plt.stem(t, signal, basefmt=" ")
            plt.title("Discrete-Time Sinusoidal Signal")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
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
        messagebox.showerror(
            "Input Error", "Please enter valid numeric values for all parameters."
        )
        return


def SaveWaveData(
    t, signal, t_discrete, discreteSignal, SamplingRate, IsPeriodic, SignalType
):
    """Saves the generated wave data to a file."""
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", ".txt"), ("All files", ".*")],
        title="Save Wave Data",
    )
    if file_path:
        try:
            with open(file_path, "w") as file:
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
                    plt.plot(time, amplitude, label="Time Domain Signal", color="blue")
                    plt.title("(Continuous)")
                    plt.xlabel("time (s)")
                    plt.ylabel("Amplitude")
                    plt.grid(True)
                    plt.legend()
                    plt.show()
                    # Plot discrete time-domain signal
                    plt.figure(figsize=(10, 5))
                    plt.stem(time, amplitude, basefmt=" ")
                    plt.title("(Discrete)")
                    plt.xlabel("Sample Index")
                    plt.ylabel("Amplitude")
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
                    plt.plot(
                        frequency,
                        amplitude,
                        label="Frequency Domain Signal",
                        color="blue",
                    )
                    plt.title("(Continuous)")
                    plt.xlabel("Frequency (Hz)")
                    plt.ylabel("Amplitude")
                    plt.grid(True)
                    plt.legend()
                    plt.show()
                    # Plot phase shift
                    plt.figure(figsize=(10, 5))
                    plt.plot(frequency, phase_shift, label="Phase Shift", color="green")
                    plt.title("Frequency Domain Phase Shift")
                    plt.xlabel("Frequency (Hz)")
                    plt.ylabel("Phase Shift (radians)")
                    plt.grid(True)
                    plt.legend()
                    plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {e}")


def CompareTask2():
    file1 = filedialog.askopenfilename(title="Select the first file")
    file2 = filedialog.askopenfilename(title="Select the second file")

    if file1 and file2:
        expected_samples = []
        with open(file1, "r") as f:
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
        with open(file1, "r") as f1:
            for i in range(3):
                line = f1.readline()
                if line:
                    value = float(line.strip())
                    sums[i] += value / 2
            for line in f1:
                index, value = line.split()
                dict1[int(index)] = float(value)
        with open(file2, "r") as f2:
            for i in range(3):
                line = f2.readline()
                if line:
                    value = float(line.strip())
                    sums[i] += value / 2
            for line in f2:
                index, value = line.split()
                dict2[int(index)] = float(value)
        with open(outputFile, "w") as out_file:
            out_file.write(f" {int(sums[0])}\n")
            out_file.write(f" {int(sums[1])}\n")
            out_file.write(f" {int(sums[2])}\n")
            all_indices = set(dict1.keys()).union(set(dict2.keys()))
            for index in sorted(all_indices):
                value1 = dict1.get(index, 0)
                value2 = dict2.get(index, 0)
                result = value1 + value2
                out_file.write(f"{index} {result}\n")
        messagebox.showinfo(
            "Operation completed successfully", f"Results saved in {outputFile}"
        )
    except FileNotFoundError:
        messagebox.showerror("Error", "One of the files is missing.")
    except Exception as e:
        messagebox.showerror("An error occurred", str(e))


def ChooseFileForAddition():
    file1 = filedialog.askopenfilename(title="Select the first file")
    file2 = filedialog.askopenfilename(title="Select the second file")
    outputFile = filedialog.asksaveasfilename(
        title="Select output file", defaultextension=".txt"
    )

    if file1 and file2 and outputFile:
        AddFile(file1, file2, outputFile)
    else:
        messagebox.showwarning("Warning", "You must select all files.")


def SubtractFile(file1, file2, outputFile):
    try:
        dict1 = {}
        dict2 = {}
        sums = [0, 0, 0]
        with open(file1, "r") as f1:
            for i in range(3):
                line = f1.readline()
                if line:
                    value = float(line.strip())
                    sums[i] += value / 2
            for line in f1:
                index, value = line.split()
                dict1[int(index)] = float(value)
        with open(file2, "r") as f2:
            for i in range(3):
                line = f2.readline()
                if line:
                    value = float(line.strip())
                    sums[i] += value / 2
            for line in f2:
                index, value = line.split()
                dict2[int(index)] = float(value)
        with open(outputFile, "w") as out_file:
            out_file.write(f" {int(sums)[0]}\n")
            out_file.write(f" {int(sums[1])}\n")
            out_file.write(f" {int(sums[2])}\n")
            all_indices = set(dict1.keys()).union(set(dict2.keys()))
            for index in sorted(all_indices):
                value1 = dict1.get(index, 0)
                value2 = dict2.get(index, 0)
                result = value1 - value2
                out_file.write(f"{index} {result}\n")
        messagebox.showinfo(
            "Operation completed successfully", f"Results saved in {outputFile}"
        )
    except FileNotFoundError:
        messagebox.showerror("Error", "One of the files is missing.")
    except Exception as e:
        messagebox.showerror("An error occurred", str(e))


def ChooseFileForSubtraction():
    file1 = filedialog.askopenfilename(title="Select the first file")
    file2 = filedialog.askopenfilename(title="Select the second file")
    outputFile = filedialog.asksaveasfilename(
        title="Select output file", defaultextension=".txt"
    )

    if file1 and file2 and outputFile:
        SubtractFile(file1, file2, outputFile)
    else:
        messagebox.showwarning("Warning", "You must select all files.")


def Multiplication(constEN):
    try:
        file1 = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        const = int(constEN.get())
        signal_dic = {}
        if not file1:
            messagebox.showerror("Error", "No file selected.")
            return

        with open(file1, "r") as file:
            lines = file.readlines()

        signal_type = int(lines[0].strip())
        isPeriodic = int(lines[1].strip())
        sampling_number = int(lines[2].strip())

        for line in lines[3:]:
            index, value = line.split()
            signal_dic[int(index)] = int(value)
        output_signal = {key: value * const for key, value in signal_dic.items()}
        outputFile = filedialog.asksaveasfilename(filetypes=[("Text files", "*.txt")])
        if outputFile:
            with open(outputFile, "w") as out_file:
                out_file.write(f"{signal_type} \n")
                out_file.write(f"{isPeriodic} \n")
                out_file.write(f"{sampling_number} \n")
                for key, value in output_signal.items():
                    out_file.write(f"{key} {value} \n")

            messagebox.showinfo("Success", "Output signal saved successfully.")
        else:
            messagebox.showwarning("Warning", "No output file selected.")
    except ValueError:
        messagebox.showerror("Error", "Please provide a valid integer or file.")
    except Exception as e:
        messagebox.showerror("An error occurred", str(e))


def Squaring():
    try:
        file1 = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])

        signal_dic = {}
        if not file1:
            messagebox.showerror("Error", "No file selected.")
            return

        with open(file1, "r") as file:
            lines = file.readlines()

        signal_type = int(lines[0].strip())
        isPeriodic = int(lines[1].strip())
        sampling_number = int(lines[2].strip())

        for line in lines[3:]:
            index, value = line.split()
            signal_dic[int(index)] = int(value)

        output_signal = {key: value * value for key, value in signal_dic.items()}

        outputFile = filedialog.asksaveasfilename(filetypes=[("Text files", "*.txt")])

        if outputFile:
            with open(outputFile, "w") as out_file:
                out_file.write(f"{signal_type} \n")
                out_file.write(f"{isPeriodic} \n")
                out_file.write(f"{sampling_number} \n")
                for key, value in output_signal.items():
                    out_file.write(f"{key} {value} \n")
            messagebox.showinfo("Success", "Output signal saved successfully.")
        else:
            messagebox.showwarning("Warning", "No output file selected.")
    except ValueError:
        messagebox.showerror("Error", "Please provide a valid integer or file.")
    except Exception as e:
        messagebox.showerror("An error occurred", str(e))


def Normalization(file1, outputFile, cmbo2):
    try:
        dict1 = {}
        sums = [0, 0, 0]
        with open(file1, "r") as f1:
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
        with open(outputFile, "w") as out_file:
            out_file.write(f" {int(sums[0])}\n")
            out_file.write(f" {int(sums[1])}\n")
            out_file.write(f" {int(sums[2])}\n")
            all_indices = set(dict1.keys())
            if cmbo2.get() == "from 0 to 1":
                for index in sorted(all_indices):
                    value1 = dict1.get(index, 0)
                    result = (value1 - min_value) / (max_value - min_value)
                    result = round(result, 3)
                    out_file.write(f"{index} {result}\n")
            else:
                for index in sorted(all_indices):
                    value1 = dict1.get(index, 0)
                    result = (2 * (value1 - min_value) / (max_value - min_value)) - 1
                    result = round(result, 3)
                    out_file.write(f"{index} {result}\n")

        messagebox.showinfo(
            "Operation completed successfully", f"Results saved in {outputFile}"
        )
    except FileNotFoundError:
        messagebox.showerror("Error", "One of the files is missing.")
    except Exception as e:
        messagebox.showerror("An error occurred", str(e))


def ChooseFileForNormalization(cmbo2):
    file1 = filedialog.askopenfilename(title="Select the first file")
    outputFile = filedialog.asksaveasfilename(
        title="Select output file", defaultextension=".txt"
    )

    if file1 and outputFile:
        Normalization(file1, outputFile, cmbo2)
    else:
        messagebox.showwarning("Warning", "You must select all files.")


def Accumulation(file1, outputFile):
    try:
        dict1 = {}
        sums = [0, 0, 0]
        with open(file1, "r") as f1:
            for i in range(3):
                line = f1.readline()
                if line:
                    value = float(line.strip())
                    sums[i] += value
            for line in f1:
                index, value = line.split()
                dict1[int(index)] = float(value)

        result = 0
        with open(outputFile, "w") as out_file:
            out_file.write(f" {int(sums[0])}\n")
            out_file.write(f" {int(sums[1])}\n")
            out_file.write(f" {int(sums[2])}\n")
            all_indices = set(dict1.keys())
            for index in sorted(all_indices):
                value1 = dict1.get(index, 0)
                result = result + value1
                out_file.write(f"{index} {result}\n")
        messagebox.showinfo(
            "Operation completed successfully", f"Results saved in {outputFile}"
        )
    except FileNotFoundError:
        messagebox.showerror("Error", "One of the files is missing.")
    except Exception as e:
        messagebox.showerror("An error occurred", str(e))


def ChooseFileForAccumulation():
    file1 = filedialog.askopenfilename(title="Select the first file")
    outputFile = filedialog.asksaveasfilename(
        title="Select output file", defaultextension=".txt"
    )

    if file1 and outputFile:
        Accumulation(file1, outputFile)
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
        binaryLevel = format(levelIndex, "0" + str(int(np.log2(numberOfLevels))) + "b")
        level_In_Binary.append(binaryLevel)
        levelIndices.append(levelIndex + 1)

    return np.array(quantizedValues), midpoints, level_In_Binary, levelIndices


def SaveQuantizeData1(
    filename,
    skippedRows,
    quantizedValues,
    midpoints,
    level_In_Binary,
    levelIndices,
    originalSignal,
):
    with open(filename, "w") as f:
        for row in skippedRows:
            f.write(f"{row}")

        # Calculate the difference between the original signal and quantized values
        differences = quantizedValues - originalSignal

        for quantized_value, binary, level_index, diff in zip(
            quantizedValues, level_In_Binary, levelIndices, differences
        ):
            f.write(f"{level_index} {binary} {quantized_value:.3f} {diff:.3f}\n")


def SaveQuantizeData(
    filename, skippedRows, quantizedValues, level_In_Binary, originalSignal
):
    with open(filename, "w") as f:
        for row in skippedRows:
            f.write(f"{row}")

        # Calculate the difference between the original signal and quantized values
        differences = quantizedValues - originalSignal

        for binary, quantized_value, diff in zip(
            level_In_Binary, quantizedValues, differences
        ):
            f.write(f"{binary} {quantized_value:.2f} \n")


def SelectFile1(entreFile1):
    filePath = filedialog.askopenfilename(
        title="Select Input File",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
    )
    if filePath:
        entreFile1.delete(0, END)
        entreFile1.insert(0, filePath)


def SelectFile2(entreFile2):
    filePath = filedialog.asksaveasfilename(
        title="Select Output File",
        defaultextension=".txt",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
    )
    if filePath:
        entreFile2.delete(0, END)
        entreFile2.insert(0, filePath)


def ProcessFilesForQuantization(entreFile1, entreFile2, combo_choice, entry_choice):
    inputFile = entreFile1.get()
    outputFile = entreFile2.get()
    choice = combo_choice.get()
    choiceValue = entry_choice.get()

    if not inputFile or not outputFile or not choiceValue:
        messagebox.showwarning(
            "Warning", "Please select both files and enter levels or bits."
        )
        return

    try:
        with open(inputFile, "r") as f:
            lines = f.readlines()

        skippedRows = lines[:3]
        signal = np.array([float(line.split()[1].strip()) for line in lines[3:]])

        if choice == "Number of bits":
            numberOfBits = int(choiceValue)
            numberOfLevels = 2**numberOfBits
            quantizedValues, midpoints, level_In_Binary, levelIndices = QuantizeSignal(
                signal, numberOfLevels
            )
            SaveQuantizeData(
                outputFile, skippedRows, quantizedValues, level_In_Binary, signal
            )
        else:
            numberOfLevels = int(choiceValue)
            quantizedValues, midpoints, level_In_Binary, levelIndices = QuantizeSignal(
                signal, numberOfLevels
            )
            SaveQuantizeData1(
                outputFile,
                skippedRows,
                quantizedValues,
                midpoints,
                level_In_Binary,
                levelIndices,
                signal,
            )

        messagebox.showinfo(
            "Success", f"Quantization complete!\nOutput saved to: {outputFile}"
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))


def CompareTask3(cmbo):
    file1 = filedialog.askopenfilename(title="Select the first file")
    file2 = filedialog.askopenfilename(title="Select the second file")
    if file1 and file2:
        if cmbo.get() == "Number of bits":
            expected_samples = []
            expected_index = []
            with open(file1, "r") as f:
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

            QuantizationTest1(file2, expected_index, expected_samples)
        else:
            expected_samples = []
            expected_index = []
            IntervalIndices = []
            SampledError = []
            with open(file1, "r") as f:
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

            QuantizationTest2(
                file2, IntervalIndices, expected_index, expected_samples, SampledError
            )

    else:
        messagebox.showwarning("Warning", "You must select all files.")


# Task 4
def ReadFrequencyComponents(file_path):
    frequencyComponents = []
    skippedRows = []
    try:
        with open(file_path, "r") as file:
            for i, line in enumerate(file):
                if i < 3:
                    skippedRows.append(line.strip())  # Save the first 3 rows
                    continue
                line = line.strip().replace("f", "")  # Remove unwanted 'f' characters
                try:
                    magnitude, phase = map(float, line.split(","))  # Convert to float
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
    inputFile = entry_file1.get()
    outputFile = entry_file2.get()

    if not inputFile or not outputFile:
        return "Please select both input and output files."

    # Read frequency components
    skippedRows, frequencyComponents = ReadFrequencyComponents(inputFile)

    if frequencyComponents is None:
        return "Error reading frequency components."

    # Convert polar to rectangular components
    convertComponents = Convert(frequencyComponents)

    # Apply IDFT to get the time-domain signal
    reconstructed_signal = IDFTConvert(convertComponents)

    # Write to output file
    with open(outputFile, "w") as out_file:
        # Write the first three rows (header)
        for line in skippedRows:
            out_file.write(line + "\n")

        # Write the index, magnitude
        for index, value in enumerate(reconstructed_signal):
            realPart = round(value.real, 2)
            imagPart = round(value.imag, 2)
            magnitude = int(
                round(np.sqrt(realPart**2 + imagPart**2), 2)
            )  # Calculate magnitude
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
        with open(file_path, "r") as file:
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

        signalData = np.array(data)
        N = len(signalData)

        frequencyComponents = []
        for k in range(N):
            realSum = 0
            imagSum = 0
            for n in range(N):
                angle = -2 * np.pi * k * n / N
                realSum += signalData[n] * np.cos(angle)
                imagSum += signalData[n] * np.sin(angle)
            frequencyComponents.append(complex(realSum, imagSum))

        # All N frequencies
        frequencies = np.arange(N) * (sampling_frequency / N)
        amplitudes = np.abs(frequencyComponents)
        phases = np.angle(frequencyComponents)
        theta = (2 * 3.14) / (N * (1 / sampling_frequency))

        # Plotting frequency vs amplitude and frequency vs phase
        plt.figure(figsize=(8, 6))

        result_list = [theta * i for i in range(1, N + 1)]
        print(result_list)

        plt.subplot(2, 1, 1)
        plt.stem(result_list, amplitudes, basefmt=" ")
        plt.title("Frequency vs amplitudes ")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Adjusted Phase (radians)")
        plt.grid()
        # Frequency vs Phase
        plt.subplot(2, 1, 2)
        plt.stem(result_list, phases, basefmt=" ")
        plt.title("Frequency vs Adjusted Phase ")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Adjusted Phase (radians)")
        plt.grid()

        plt.tight_layout()
        plt.show()

        # Save output in the specified format
        with open(output_path, "w") as outfile:
            outfile.write(f"{signal_type} \n")
            is_periodic = 1
            outfile.write(f"{is_periodic} \n")
            outfile.write(f"{N1} \n")

            for i in range(N):
                amplitude_str = (
                    f"{int(amplitudes[i])}"
                    if amplitudes[i].is_integer()
                    else f"{amplitudes[i]:.14f}f"
                )
                phase_str = (
                    f"{int(phases[i])}"
                    if phases[i].is_integer()
                    else f"{phases[i]:.14f}f"
                )
                outfile.write(f"{amplitude_str} {phase_str}\n")

        return f"DFT results saved to {output_path}."

    except ValueError:
        return "Invalid data or sampling frequency."
    except Exception as e:
        return f"An error occurred: {e}"


# Function to check combo box selection and call appropriate function
def check_and_process(combo_box, entry_file, entry_outputFile, entry_sampling_freq):
    selected_option = combo_box.get()
    if selected_option == "DFT":
        output_path = entry_outputFile.get()
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
        result = ProcessFilesForiDFT(entry_file, entry_outputFile)
        return result


def CompareTask4():
    file1 = filedialog.askopenfilename(title="Select the first file")
    file2 = filedialog.askopenfilename(title="Select the second file")

    if file1 and file2:
        samplesfile1 = []
        indexfile1 = []
        samplesfile2 = []
        indexfile2 = []

        # Read first file
        with open(file1, "r") as f:
            for _ in range(3):
                f.readline()  # Skip first three lines
            for line in f:
                L = line.strip().replace("f", "")  # Remove 'f' before processing
                if len(L.split()) == 2:
                    V1, V2 = map(float, L.split())
                    indexfile1.append(V1)
                    samplesfile1.append(V2)

        # Read second file
        with open(file2, "r") as f:
            for _ in range(3):
                f.readline()  # Skip first three lines
            for line in f:
                L = line.strip().replace("f", "")  # Remove 'f' before processing
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


# task 5


def sharpening(input, output):
    input = input.get()
    output = output.get()

    skippedRows = []
    Signal = []
    firstDerivative = []
    secondDerivative = []

    try:
        with open(input, "r") as f:
            # Read the first three rows (just values)
            for i in range(3):
                line = f.readline().strip()
                skippedRows.append(int(line))

            # Read the remaining rows (index-value pairs)
            for line in f:
                index, value = line.strip().split()
                
                # Remove the 'f' character and convert the value to float or integer
                value = float(value.rstrip('f'))  # Remove 'f' and convert to float
                
                # Append the cleaned data to Signal
                Signal.append((int(index), value))

            # Calculate the first and second derivatives
            for i in range(len(Signal)):
                index, value = Signal[i]

                # First Derivative: (f[i] - f[i-1])
                if i == 0:
                    # Handle first element (no previous element)
                    firstDerivativeValue = value
                else:
                    firstDerivativeValue = Signal[i][1] - Signal[i - 1][1]
                firstDerivative.append((index, firstDerivativeValue))

                # Second Derivative: (f[i+1] + f[i-1] - 2*f[i])
                if i == 0 or i == len(Signal) - 1:
                    # Handle first and last elements (set second derivative to 0 or ignore for boundary points)
                    secondDerivativeValue = 0  # Set to 0 for boundary points
                else:
                    secondDerivativeValue = (
                        Signal[i + 1][1] + Signal[i - 1][1] - 2 * value
                    )
                secondDerivative.append((index, secondDerivativeValue))


        # Plotting signal, first and second derivatives
        plt.figure(figsize=(8, 6))

        # Signal Plot (Continuous Line)
        plt.subplot(3, 1, 1)
        indices, values = zip(*Signal)
        plt.plot(indices, values, label="Signal", color='b')  # Continuous line for signal
        plt.title("Signal")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.grid()
        plt.legend()

        # First Derivative Plot (Continuous Line)
        plt.subplot(3, 1, 2)
        indices, values = zip(*firstDerivative)
        plt.plot(indices, values, label="First Derivative", color='g')  # Continuous line for first derivative
        plt.title("First Derivative")
        plt.xlabel("Index")
        plt.ylabel("First Derivative Value")
        plt.grid()
        plt.legend()

        # Second Derivative Plot (Continuous Line)
        plt.subplot(3, 1, 3)
        indices, values = zip(*secondDerivative)
        plt.plot(indices, values, label="Second Derivative", color='r')  # Continuous line for second derivative
        plt.title("Second Derivative")
        plt.xlabel("Index")
        plt.ylabel("Second Derivative Value")
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()


        # Save the modified signal to a new file
        with open(output, "w") as f:
            # Write the first three values unchanged (no indices)
            for value in skippedRows:
                f.write(f"{value}\n")
            # Write the index, original signal, first derivative, and second derivative on the same row
            for i in range(len(Signal)):
                index = Signal[i][0]
                first_derivative_value = firstDerivative[i][1]
                second_derivative_value = secondDerivative[i][1]
                # Write all values on the same line
                f.write(f"{int(index)} {int(first_derivative_value)} {int(second_derivative_value)}\n")

        messagebox.showinfo("successful", "Sharpening completed successfully.")
    
    except FileNotFoundError:
        print(f"Error: File {input} not found.")
    except ValueError:
        print("Error: Non-numeric data found in the file.")
    
    return skippedRows, Signal, firstDerivative, secondDerivative




def DCT(input, output, m):
    input = input.get()
    output = output.get()
    m = int(m.get()) 
    try:
        with open(input, "r") as file:
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

        signalData = np.array(data)
        N = len(signalData)
        y = np.zeros(N)
        factor = np.sqrt(2 / N)

        for k in range(N):
            sum_val = 0
            for n in range(N):
                sum_val += signalData[n] * np.cos((np.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1)   )
            y[k] = factor * sum_val
        
        with open(output, "w") as outfile:
            outfile.write(f"{signal_type} \n")  
            is_periodic = 1 
            outfile.write(f"{is_periodic} \n")
            outfile.write(f"{N1} \n") 
            
            for value in y[:m]: 
                outfile.write(f"0 {value}\n")
        
        messagebox.showinfo("Successful", "DCT completed successfully.")
    except Exception as e:
        return f"An error occurred: {e}"


def chooseoperation(inputFile, outputFile, cmbo,m):
    if cmbo.get() == "Sharpening":
        sharpening(inputFile, outputFile)
    else:
        DCT(inputFile, outputFile,m)


from tkinter import filedialog, messagebox

def CompareTask5(m, cmbo):
    # Initialize exceptedIndex and expected_samples at the start to avoid unbound error
    exceptedIndex = []
    expected_samples = []

    if cmbo.get() == "DCT":
        # DCT comparison: Select two files
        file1 = filedialog.askopenfilename(title="Select the first file")
        file2 = filedialog.askopenfilename(title="Select the second file")
        
        if file1 and file2:
            # Read data from file1
            with open(file1, "r") as f:
                for _ in range(3):  # Skip first three lines
                    f.readline()
                for line in f:
                    L = line.strip().split()
                    if len(L) == 2:
                        V1 = int(L[0])
                        V2 = float(L[1])
                        expected_samples.append(V2)
                        exceptedIndex.append(V1)
                    else:
                        break
            # Call DCTSignalCompare (ensure this function is defined elsewhere)
            DCTSignalCompare(file2, exceptedIndex, expected_samples, m)
        else:
            messagebox.showwarning("Warning", "You must select both files.")
    
    else:  # Sharpening comparison: Select one file
        file1 = filedialog.askopenfilename(title="Select the first file")
        
        if file1:
            # Read data from the selected file
            with open(file1, "r") as f:
                for _ in range(3):  # Skip first three lines
                    f.readline()
                for line in f:
                    L = line.strip().split()
                    if len(L) == 3:
                        V1 = int(L[1])
                        V2 = float(L[2])
                        exceptedIndex.append(V1)
                        expected_samples.append(V2)
                    else:
                        break
            # Call SharpeningCompare (ensure this function is defined elsewhere)
            exceptedIndex = exceptedIndex[:-1]  # Skip last row in V1
            expected_samples = expected_samples[:-2]  # Skip last two rows in V2
            SharpeningCompare(exceptedIndex, expected_samples)
        else:
            messagebox.showwarning("Warning", "You must select a file.")




# Task 6
def Readsignal(inputFile):
    skippedRows = []
    foldingSignal = []

    try:
        with open(inputFile, "r") as f:
            # Read the first three rows (just values)
            for i in range(3):
                line = f.readline().strip()
                skippedRows.append(int(line))

            # Read the remaining rows (index-value pairs)
            for line in f:
                index, value = line.strip().split()
                foldingSignal.append((int(index), int(value)))

    except FileNotFoundError:
        print(f"Error: File {inputFile} not found.")
    except ValueError:
        print("Error: Non-numeric data found in the file.")

    return skippedRows, foldingSignal


# Function to reverse the values (Folding)
def ReverseValues(signal):
    values = [value for _, value in signal]
    reversedValues = values[::-1]

    # Combine the original indices with the reversed values
    reversedSignal = [(signal[i][0], reversedValues[i]) for i in range(len(signal))]

    return reversedSignal


# Function to save the modified signal to a new file
def SaveFile(ouputFile, skippedRows, modifiedSignal):
    try:
        with open(ouputFile, "w") as f:
            # Write the first three values unchanged (no indices)
            for value in skippedRows:
                f.write(f"{value}\n")

            # Write the remaining rows with the modified values
            for index, value in modifiedSignal:
                f.write(f"{index} {value}\n")
    except Exception as e:
        messagebox.showerror("Error", f"Error saving the result to {ouputFile}: {e}")


# Function to shift the values based on the shift amount
def Shifting(inputFile, shifting, outputFile=None):
    newData = []
    with open(inputFile, "r") as file:
        lines = file.readlines()

        # First 3 lines are assumed to have only values and are kept for output without modification
        header_lines = lines[:3]
        data_lines = lines[3:]

        # Add the first three value-only lines to the output
        newData.extend(header_lines)

        # Process each remaining line containing index and value
        for line in data_lines:
            # Assuming the format is 'index value'
            index, value = map(int, line.split())

            # Shift the index
            if int(shifting) < 0:
                newIndex = index - shifting
            else:
                newIndex = index - shifting
            # Store the new index-value pair
            newData.append(f"{newIndex} {value}\n")

    # Write to the output file if provided
    if outputFile:
        with open(outputFile, "w") as outputFile:
            outputFile.writelines(newData)

    # Return the new lines if no output file is provided
    return newData if not outputFile else None


# Function for handling the process choice (Folding or Shifting)
def ChooseProccess(inputFile, outputFile, cmbo2, value):
    shiftValue = int(value.get())  # Get the shift value as an integer

    if cmbo2.get() == "Folding":
        outputPath = outputFile.get()
        inputPath = inputFile.get()
        skippedRows, foldingSignal = Readsignal(inputPath)
        reversedSignal = ReverseValues(foldingSignal)
        SaveFile(outputPath, skippedRows, reversedSignal)
        messagebox.showinfo("Successful", "Folding done successfully!")

    elif cmbo2.get() == "Shifting":
        outputPath = outputFile.get()
        inputPath = inputFile.get()

        # Correct the order of arguments in the Shifting function
        Shifting(inputPath, shiftValue, outputPath)
        messagebox.showinfo("Successful", "Shifting done successfully!")
    else:
        shiftValue = -int(value.get())
        outputPath = outputFile.get()
        inputPath = inputFile.get()
        skippedRows, foldingSignal = Readsignal(inputPath)
        reversedSignal = ReverseValues(foldingSignal)
        SaveFile(outputPath, skippedRows, reversedSignal)
        Shifting(outputPath, shiftValue, outputPath)
        messagebox.showinfo("Successful", "Folding and Shifting done successfully!")


def CompareTask6():
    file1 = filedialog.askopenfilename(title="Select the first file")
    file2 = filedialog.askopenfilename(title="Select the second file")

    if file1 and file2:
        expected_samples = []
        exceptedIndex = []
        with open(file1, "r") as f:
            for _ in range(3):
                f.readline()
            for line in f:
                L = line.strip()
                if len(L.split()) == 2:
                    L = L.split()
                    V1 = int(L[0])
                    V2 = float(L[1])
                    expected_samples.append(V2)
                    exceptedIndex.append(V1)
                else:
                    break

        ShiftFoldSignal(file2, exceptedIndex, expected_samples)
    else:
        messagebox.showwarning("Warning", "You must select all files.")
# Task 7
def GetMinIndex(file1, file2):
    # Load data from both files (skip the first 3 rows)
    data1 = np.loadtxt(file1, skiprows=3)
    data2 = np.loadtxt(file2, skiprows=3)
    
    # Extract indices from the first column
    indices1 = data1[:, 0]
    indices2 = data2[:, 0]
    
    # Return the minimum index from both files
    return min(min(indices1), min(indices2))

def Convolve(x, h, min_index):
    # Perform manual convolution
    
    # Ensure n_min and n_max are integers
    n_min = min(range(len(x))) + min(range(len(h)))
    n_max = max(range(len(x))) + max(range(len(h)))

    y = []
    for n in range(n_min, n_max + 1):
        sum_val = 0
        for k in range(len(x)):
            if 0 <= n - k < len(h):  # Ensure indices are within bounds of h
                sum_val += x[k] * h[n - k]
        
        result_index =  min_index + n  # Adjust result index relative to min_index
        y.append((int(result_index), int(sum_val)))  # Store adjusted index-value pair
    
    return y

def MovingAverage(filePath ,windowSize):
    indices, samples = [], []
    with open(filePath, 'r') as f:
        for _ in range(3):  # Skip the first three lines
            f.readline()
        for line in f:
            values = line.strip().split()
            if len(values) == 2:
                indices.append(int(values[0]))
                samples.append(float(values[1]))
    windowSize = int(windowSize)
    
    if windowSize < 1 or windowSize > len(samples):
        raise ValueError("Window size must be a positive integer smaller than or equal to the length of the samples.")   
    result = np.zeros(len(samples) - windowSize + 1)
    
    # Compute the moving average
    for i in range(len(result)):
        result[i] = np.sum(samples[i:i+windowSize]) / windowSize
    return result


        
def RemoveDcInFrequencyDomain(inputFile, outputFile):
    try:
        # Step 1: Read input file
        with open(inputFile, "r") as file:
            lines = file.readlines()
            data = []
            skippedRows = lines[:3]  # Save the first 3 header lines
            for line in lines[3:]:
                parts = line.strip().split()
                if len(parts) == 2:
                    time, amplitude = map(float, parts)
                    data.append(amplitude)

        signalData = np.array(data)
        N = len(signalData)
        frequencyComponents = []

        # Step 2: Perform DFT
        for k in range(N):
            realSum = 0
            imagSum = 0
            for n in range(N):
                angle = -2 * np.pi * k * n / N
                realSum += signalData[n] * np.cos(angle)
                imagSum += signalData[n] * np.sin(angle)
            frequencyComponents.append(complex(realSum, imagSum))

        # Step 3: Modify the first frequency component
        frequencyComponents[0] = 0  # Set both real and imaginary parts to 0

        # Step 4: Perform IDFT
        timeSignal = np.zeros(N, dtype=complex)
        for n in range(N):
            for k in range(N):
                timeSignal[n] += frequencyComponents[k] * np.exp(1j * (2 * np.pi / N) * k * n)
        timeSignal /= N  # Normalize

        # Step 5: Save the reconstructed signal to the output file
        with open(outputFile, "w") as out_file:
            # Write the header lines
            for line in skippedRows:
                out_file.write(line)

            # Write the reconstructed signal
            for index, value in enumerate(timeSignal):
                real_part = value.real
                imag_part = value.imag
                magnitude = np.sqrt(real_part**2 + imag_part**2)
                # Negate magnitude if imaginary part is negative
                if real_part < 0:
                    magnitude = -magnitude
                out_file.write(f"{index} {magnitude:.3f}\n")  # Format magnitude to 3 decimal places

        # return "Processing complete. Output saved."
    except Exception as e:
        return f"An error occurred: {e}"
def RemoveDcInTimeDomain(inputFile, outputFile):
    try:
        expected_indices = []
        expected_samples = []
        skippedRows = []
        
        # Reading the input file and extracting the necessary information
        with open(inputFile, 'r') as f:
            for i in range(3):  # Read first 3 header lines
                line = f.readline().strip()
                skippedRows.append(line)  # Add the header to the skippedRows
            for line in f:  # Read the actual signal data
                L = line.strip()
                if len(L.split()) == 2:
                    L = L.split()
                    V1 = int(L[0])  # The index value
                    V2 = float(L[1])  # The sample value
                    expected_indices.append(V1)
                    expected_samples.append(V2)
                else:
                    break
        
        # Calculate the mean of the samples
        mean = np.mean(expected_samples)
        N = len(expected_samples)
        
        # Remove DC component by subtracting the mean from each sample
        expected_samples = [sample - mean for sample in expected_samples]
        
        # Saving the modified signal to the output file
        with open(outputFile, "w") as out_file:
            # Write the header lines
            for line in skippedRows:
                out_file.write(line + "\n")

            # Write the modified signal (after removing the DC component)
            for index, sample in zip(expected_indices, expected_samples):
                out_file.write(f"{index} {sample:.3f}\n")  # Format the sample value to 3 decimal places
        
        return "Processing complete. Output saved."
    
    except Exception as e:
        return f"An error occurred: {e}"
    
def Correlation(input1,input2,output):

    r1 = []
    r2 = []
    shifted_r2 = []

    try:
        
        with open(input1, "r") as file1:
            lines = file1.readlines()
            
            skippedRows = lines[:3]  # Save the first 3 header lines
            for line in lines[3:]:
                parts = line.strip().split()
                if len(parts) == 2:
                    index1, amplitude = map(float, parts)
                    r1.append(amplitude)  
            
        with open(input2, "r") as file2:
            lines = file2.readlines()
            
            
            for line in lines[3:]:
                parts = line.strip().split()
                if len(parts) == 2:
                    index2, amplitude = map(float, parts)
                    r2.append(amplitude) 
            

        N = len(r1)  # Length of the signals
        r12 = []  # To store raw cross-correlation values
        P12 = []  # To store normalized cross-correlation values

        # Compute power (sum of squares) of r1 and X2
        power_r1 = sum(x**2 for x in r1)
        power_r2 = sum(x**2 for x in r2)

        # Loop over shifts (j)
        for j in range(N):
            # Perform cyclic shift on r2
            shifted_r2 = r2[j:] + r2[:j]

            # Compute r12(j)
            r12_j = sum(r1[n] * shifted_r2[n] for n in range(N)) / N
            r12.append(r12_j)

            # Compute normalized P12(j)
            P12_j = r12_j / ((power_r1 / N)**0.5 * (power_r2 / N)**0.5)
            P12.append(P12_j)
        
        with open(output,"w") as out:
            for line in skippedRows:
                out.write(line)
            
            for index1,value in enumerate(P12):
                out.write(f"{index1} {value:.8f}\n")
                

        messagebox.showinfo("Successful", "Correlation done successfully!")


    except Exception as e:
        return f"An error occurred: {e}"
def ProcessConvolution(input1, input2, output, cmbo,value):
    file1 = input1.get()
    file2 = input2.get()
    outputFile = output.get()  
    if cmbo.get() == "Convolution":        
            # Read the first three rows as comments
        with open(file1, "r") as f:
            first_three_rows_file1 = [next(f).strip() for _ in range(3)]

        with open(file2, "r") as f:
            first_three_rows_file2 = [next(f).strip() for _ in range(3)]

        # Load signals (skip the first 3 rows and extract values only)
        data1 = np.loadtxt(file1, skiprows=3)
        data2 = np.loadtxt(file2, skiprows=3)
        
        # Extract indices and signal values (assuming two columns: index and value)
        indices1 = data1[:, 0]  # Extract indices from first column
        signal1Values = data1[:, 1]  # Extract signal values from second column
        
        indices2 = data2[:, 0]  # Extract indices from first column
        signal2Values = data2[:, 1]  # Extract signal values from second column
        
        # Get minimum index from both files
        minIndex = GetMinIndex(file1, file2)
        
        # Perform the convolution
        convolvedSignal = Convolve(signal1Values, signal2Values, minIndex)
        
        # Get the length of the convolved signal
        outputLength = len(convolvedSignal)

        # Save the result to the output file
        with open(outputFile, "w") as f:
            # Write the first 3 rows, which contain only one value per row (from the input files)
            f.write(f"0\n")
            f.write(f"0\n")
            f.write(f"{outputLength}\n")
            
            # Write the convolved signal (index-value pairs)
            for index, value in convolvedSignal:
                f.write(f"{index} {value}\n")
        
        messagebox.showinfo("Successful", "Convolution done successfully!")
    elif cmbo.get()== "Smoothing":
        windowSize = int(value.get())
        y = MovingAverage(file1, windowSize)
        outputLength = len(y)
        try:
        # Open the file in write mode
            with open(outputFile, 'w') as file:
                file.write(f"0\n")
                file.write(f"0\n")
                file.write(f"{outputLength}\n")
                for i in range( len(y)):  
                        file.write(f"{i} {y[i]:.6f}\n")     
            messagebox.showinfo("Successful", "Smoothing done successfully!")
        except Exception as e:
            print(f"Error saving the file: {e}")
    elif cmbo.get() == "Remove the DC":
        # Open the file and read the first line
        with open(file1, "r") as f:
            first_line = f.readline().strip()  # Read and strip the first line
        
        # Check the value of the first line and call the appropriate function
        if first_line == "0":
            RemoveDcInTimeDomain(file1, outputFile)
            messagebox.showinfo("Successful", "DC removed using time domain method!")
        elif first_line == "1":
            RemoveDcInFrequencyDomain(file1, outputFile)
            messagebox.showinfo("Successful", "DC removed using frequency domain method!")
        else:
            messagebox.showerror("Error", "Invalid file format. The first line must be 0 or 1.")

    elif cmbo.get()== "Correlation":
        Correlation(file1,file2,outputFile)
    else:
        messagebox.showerror("Error", "Invalid choice")





