build:
	@echo "Building Parallel Program - Sequence Alignment"

	@mpicxx -fopenmp -c sequence_alignment.c -o sequence_alignment.o
	@nvcc -I./inc -Xcompiler -fopenmp -c cudaCalcuations.cu -o cudaCalcuations.o
	@mpicxx -fopenmp -o runSAC sequence_alignment.o cudaCalcuations.o  /usr/local/cuda-10/targets/aarch64-linux/lib/libcudart_static.a -ldl -lrt

clean:
	@echo "Cleaning compilation files"
	@rm -f *.o ./runSAC

run:
	@echo "Running Program - Sequence Alignment"
	@mpiexec -np 2 ./runSAC < input.txt

runOn2:
	@echo "Running Program - Sequence Alignment - on 2 machines"
	@mpiexec -np 2 -machinefile  mf -map-by node ./runSAC