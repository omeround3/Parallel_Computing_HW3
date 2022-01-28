# Sequence Alignment using MPI, OpenMP and CUDA

_Parallel and Distributed Computation_


## Project Overview

The following project solves the sequence alignment problem, which deals with comparisons between series of characters. That is defined by an **alignment score** which is determined by a comparison between 2 given characters sequences. The higher the score, the more similar the series are considered.

## Data Structures

### **chars\_comparison**

A 2D char array, with the size of 26 X 26, for storing characters comparison. Each cell in the 2D array (matrix) represents the comparison of 2 characters from the English alphabet.

For example, the characters pair (S, A) is present in first\_type\_group. So the corresponding cell for this pair would be chars\_comparison[&#39;S&#39;-&#39;A&#39;][&#39;A&#39;-&#39;A&#39;] = &#39;%&#39;

The ASCII values of the characters are being used to deduct the corresponding letter ASCII value to match correct matrix cell (e.g., A = 65 in ASCII A – A = 65 – 65 = 0). Therefore, we only need to deduct each given character the ASCII value of A to get the right index.

### **first\_type\_groups**

Strings from the first type groups are stored in this char array. They are used to build the chars\_comparison matrix.

### **second\_type\_groups**

Strings from the second type groups are stored in this char array. They are used to build the chars\_comparison matrix.

### **weights**

An int array to store weights which are read from the standard input.

### **Payload**

A struct defined to store 2 characters sequences to be compared.

- **seq1** – A char array to store first sequence from the standard input. This is the sequence to be compared to.
- **seq2** – A char array to store a sequence to be compared to seq1. Read from standard input.
- **len** – An int value to store the length of seq2.
- **max\_offset** – An int value to store the max offset between seq1 and seq2. Defined by

### **Score**

A struct defined to store an alignment score, specially and alignment with specific offset and hyphen indices.

- **offset** – An int value to store the offset of seq2. Either the best offset found, or the one to be checked.
- **hyphen\_idx** – An int value to store the mutant index (hyphen index) of seq2. Either the best mutant index found, or the one to be checked.
- **alignment\_score** – An int value to store the alignment score.
- **max\_score** – an int value to store the max score seq2 can get. Defined by the length of .

# Parallelism Algorithm

1. Master process (process 0) creates the 2D matrix (table) according to first type and second type groups strings.
2. Master process (process 0) reads from standard input and creates Payload and Score structs instances according to the number of sequences in the input.
3. Master process (process 0) broadcasts the necessary shared data to all MPI processes
4. Other processes allocate Payload and Score structs according to the shared data received from the master process.
5. Master process (process 0) broadcasts Payload and Score structs to other MPI processes.
6. Each MPI process iterates on the number of sequences read from the standard input and calculates the alignment score for the work size the process has
    1. Each process calculates the work size for a given sequence by. The remainder (if exists) is calculated with the same parameters but using modules instead of division.
    2. Each MPI process launches CUDA with the first half of his work size. The data sent to the device (GPU) is payload[i], score[i], start\_offset and end\_offset. Where **i** represents the index of the sequence.
        1. The number of blocks is determined by the length of seq2. If it is bigger the maximum number of allowed threads (1024), then an extra block will be used.
        2. The number of threads is defined by the number of allowed threads (1024).
        3. Each thread is responsible to find the optimal mutant sequence. Therefore, for each sequence offset a new kernel function is launched.
        4. Iterating the sequence offset has been parallelized with OpenMP.
    3. Each MPI process launches OpenMP with the second half of his work size. Each thread is responsible to find the optimal mutant sequence.
    4. Each process other then the master process will send its best score for a given sequence back to the master process.
7. Master process (process 0) receives best score for each sequence and compares and swaps with its results if necessary.
8. Master process (process 0) prints the results to standard output

## How to Run
In order to run his program you need an Ubuntu 20.04 / 18.04 machine with the next dependeices:
- CUDA 10 or 11
- OpenMP
- MPI
- gcc

### Makefile usage
- run `make` command from project root directory to build the solution
- run `make run` to run the project with 2 processes (number of process can be changed in the Makefile by changed the number after the `-n` arugment)
- run `make clean` to clean up the build files and executable