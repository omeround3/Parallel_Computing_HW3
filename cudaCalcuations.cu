#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <string.h>
#include "sequence_alignment.h"

__device__ int dev_strlen(char *dest) {
	int count = 0;
	char c = dest[count];
	while (c != '\0') {
		count++;
		c = dest[count];
	}
	return count;

}

__device__ char* dev_strcpy(char *dest, char *source) {
	char *ptr = dest;
	while (*source != '\0') {
		*dest = *source;
		dest++;
		source++;
	}
	*dest = '\0';
	return ptr;
}

__device__ Alignment* dev_copy(const Alignment *source, Alignment *dest) {
	dest->offset = source->offset;
	dest->score = source->score;
	dest->char_idx = source->char_idx;
	dest->char_val = source->char_val;
	return dest;
}

__device__ Alignment* dev_compare(const Data *d, Alignment *a, char *chars_comparision,
		double *weights) {
	int c1, c2;
	a->score = 0;
	for (int chr_ofst = 0; chr_ofst < d->len; ++chr_ofst) {
		c1 = d->seq1[chr_ofst + a->offset] - 'A';
		if (chr_ofst == a->char_idx) {
			c2 = a->char_val;
		} else {
			c2 = d->seq2[chr_ofst] - 'A';
		}
		switch (chars_comparision[c1 * CHARS + c2]) {
		case '$':
			a->score += weights[0];
			break;
		case '%':
			a->score -= weights[1];
			break;
		case '#':
			a->score -= weights[2];
			break;
		default:
			a->score -= weights[3];
			break;
		}
	}
	return a;
}

__device__ Alignment* dev_compare_and_swap(const Alignment *a1, Alignment *a2) {
	if (a1->score > a2->score) {
		dev_copy(a1, a2);
	}
	return a2;
}
__device__ Alignment* dev_find_offset(const Data *source, Alignment *res,
		char *chars_comparision, double *weights) {

	Alignment tmp;
	dev_copy(res, &tmp);

	for (int i = 1; i <= source->max_offset; ++i) {
		tmp.offset = i;
		dev_compare(source, &tmp, chars_comparision, weights);
		dev_compare_and_swap(&tmp, res);
	}
	return res;

}

__global__ void find_optimum(Data *data, Alignment *results, char *chars_comparision,
		double *weights, int from) {
	Alignment tmp;

// Each thread will write to element idx
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

// Each block will be responsible to an idx in seq2
	int char_idx = from + blockIdx.x;

// Each thread in block will replace to a different char (CHARS threads in total)
	int new_chr = threadIdx.x;

// "res_tmp" will hold the in/max element, "mut_tmp" is an helper
	dev_copy(&results[idx], &tmp);

// Set char to replace
	tmp.char_idx = char_idx;

// Set target char (if possible to replace)
	int c1 = data->seq2[char_idx] - 'A';
	char sign = chars_comparision[c1 * CHARS + new_chr];
	if (sign != '%' && sign != '$') {
		tmp.char_val = new_chr;
		tmp.offset = 0;
		dev_compare(data, &tmp, chars_comparision, weights);
		dev_find_offset(data, &tmp, chars_comparision, weights);
		dev_compare_and_swap(&tmp, &results[idx]);
	}
}

void cuda_free(void *ptr, cudaError_t err) {
// Free allocated memory on GPU - ArrayA
	if (cudaFree(ptr) != cudaSuccess) {
		fprintf(stderr, "Failed to free device data - %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void validate(cudaError_t err) {
	if (err != cudaSuccess) {
		fprintf(stderr, "%s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

Alignment* computeOnGPU(Data *data, Alignment *res, char *chars_comparision, double *weights, int from, int to) {
	omp_set_num_threads(THREADS);

	Alignment tmp;
	copy(res, &tmp);

	int share = to - from;
	size_t size = sizeof(Alignment) * share * CHARS;
	size_t num_of_res = share * CHARS;

// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

// Allocate results array
	Alignment *results_array = (Alignment*) malloc(size);
#pragma omp parallel for
	for (int i = 0; i < num_of_res; ++i) {
		copy(res, &results_array[i]);
	}
// Allocate memory on GPU for Source Alignment
	Data *sourceGPU;
	err = cudaMalloc((void**) &sourceGPU, sizeof(Data));
	validate(err);

// Allocate memory on GPU for Results Alignments
	Alignment *resultsGPU;
	err = cudaMalloc((void**) &resultsGPU, size);
	validate(err);

// Allocate memory on GPU for chars_comparision matrix
	char *pairsGPU;
	err = cudaMalloc((void**) &pairsGPU, CHARS * CHARS);
	validate(err);

// Allocate memory on GPU for weights array
	double *weightsGPU;
	err = cudaMalloc((void**) &weightsGPU, sizeof(double) * WEIGHTS);
	validate(err);

// Copy source from host to the GPU memory
	err = cudaMemcpy(sourceGPU, data, sizeof(Data), cudaMemcpyHostToDevice);
	validate(err);

// Copy result from host to the GPU memory
	err = cudaMemcpy(resultsGPU, results_array, size, cudaMemcpyHostToDevice);
	validate(err);

// Copy chars_comparision matrix from host to the GPU memory
	err = cudaMemcpy(pairsGPU, chars_comparision, CHARS * CHARS, cudaMemcpyHostToDevice);
	validate(err);

// Copy weights array from host to the GPU memory
	err = cudaMemcpy(weightsGPU, weights, sizeof(double) * WEIGHTS,
			cudaMemcpyHostToDevice);
	validate(err);

// Launch the Kernel
	find_optimum<<<share, CHARS>>>(sourceGPU, resultsGPU, pairsGPU, weightsGPU,from);
	err = cudaGetLastError();
	validate(err);
// Copy the  result from GPU to the host memory.
	err = cudaMemcpy(results_array, resultsGPU, size, cudaMemcpyDeviceToHost);
	validate(err);

// Find optimum in results using openmp
//	printf("SHARE: %ld\n", num_of_res);
#pragma omp parallel firstprivate(tmp) private(from,to,share)
	{
		int t_num = omp_get_thread_num();
		share = num_of_res / THREADS;
		from = t_num * share;
		to = t_num != THREADS - 1 ? (t_num + 1) * share : num_of_res;
		for (int i = from; i < to; ++i) {
			compare_and_swap(&results_array[i], &tmp);
		}
#pragma omp critical
		{
			compare_and_swap(&tmp, res);
		}
	}

// Free allocated space in GPU
	cuda_free(sourceGPU, err);
	cuda_free(resultsGPU, err);
	cuda_free(pairsGPU, err);
	cuda_free(weightsGPU, err);
	free(results_array);
	return res;
}

