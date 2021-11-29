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

__device__ Score* dev_deep_copy_score(const Score *source, Score *dest) {
	dest->offset = source->offset;
	dest->hyphen_idx = source->hyphen_idx;
	dest->alignment_score = source->alignment_score;
	dest->max_score = source->max_score;
	return dest;
}

__device__ Score* dev_compare(const Payload *d, Score *score, char *chars_comparision,
		double *weights) {
	int seq1_char, seq2_char;
	score->alignment_score = 0;
	// printf("In compare::BEFORE LOOP\t");
	for (int char_index = 0; char_index < payload->len; ++char_index) {
		/* Convert char to ABC order index */
		if (char_index == score->hyphen_idx) {
			seq1_char = payload->seq1[char_index + 1 + score->offset] - 'A';	/* skip character if index is hyphen */
		}
		else {
			seq1_char = payload->seq1[char_index + score->offset] - 'A';
		}
		seq2_char = payload->seq2[char_index] - 'A';
		/* check sign of characters */
		switch (chars_comparision[seq1_char][seq2_char]) {
			case '$':
				score->alignment_score += weights[0];
				break;
			case '%':
				score->alignment_score -= weights[1];
				break;
			case '#':
				score->alignment_score -= weights[2];
				break;
			default:
				score->alignment_score -= weights[3];
				break;
		}
	}
}

__device__ Score* dev_compare_scores_and_swap(const Score *s1, Score *s2) {
	if (s1->alignment_score > s2->alignment_score) {
		dev_deep_copy_score(s1, s2);
	}
}
__device__ Score* dev_find_optimal_offset(const Payload *source, Score *score,
		char *chars_comparision, double *weights) {

	Score tmp;
	dev_deep_copy_score(res, &tmp);

	for (int i = 1; i <= source->max_offset; ++i) {
		tmp.offset = i;
		dev_compare(source, &tmp, chars_comparision, weights);
		dev_compare_and_swap(&tmp, res);
	}
	return res;
}

__global__ void find_optimum(Payload *data, Score *results, char *chars_comparision,
		double *weights, int from) {
	Score tmp;

// Each thread will write to element idx
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

// Each block will be responsible to an idx in seq2
	int hyphen_idx = from + blockIdx.x;

// Each thread in block will replace to a different char (CHARS threads in total)
	int new_chr = threadIdx.x;

// "res_tmp" will hold the in/max element, "mut_tmp" is an helper
	dev_copy(&results[idx], &tmp);

// Set char to replace
	tmp.hyphen_idx = hyphen_idx;

// Set target char (if possible to replace)
	int c1 = data->seq2[hyphen_idx] - 'A';
	char sign = chars_comparision[c1 * CHARS + new_chr];
	if (sign != '%' && sign != '$') {
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

Score* cuda_calculation(Payload *data, Score *res, char *chars_comparision, double *weights, int from, int to) {
	omp_set_num_threads(THREADS);

	Score tmp;
	deep_copy_score(res, &tmp);

	int share = to - from;
	size_t size = sizeof(Score) * share * CHARS;
	size_t num_of_res = share * CHARS;

// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

// Allocate results array
	Score *results_array = (Score*) malloc(size);
#pragma omp parallel for
	for (int i = 0; i < num_of_res; ++i) {
		deep_copy_score(res, &results_array[i]);
	}
// Allocate memory on GPU for Source Score
	Payload *sourceGPU;
	err = cudaMalloc((void**) &sourceGPU, sizeof(Payload));
	validate(err);

// Allocate memory on GPU for Results Alignments
	Score *resultsGPU;
	err = cudaMalloc((void**) &resultsGPU, size);
	validate(err);

// Allocate memory on GPU for chars_comparision matrix
	char *pairsGPU;
	err = cudaMalloc((void**) &pairsGPU, CHARS * CHARS);
	validate(err);

// Allocate memory on GPU for weights array
	double *weightsGPU;
	err = cudaMalloc((void**) &weightsGPU, sizeof(double) * WEIGHTS_NUM);
	validate(err);

// Copy source from host to the GPU memory
	err = cudaMemcpy(sourceGPU, data, sizeof(Payload), cudaMemcpyHostToDevice);
	validate(err);

// Copy result from host to the GPU memory
	err = cudaMemcpy(resultsGPU, results_array, size, cudaMemcpyHostToDevice);
	validate(err);

// Copy chars_comparision matrix from host to the GPU memory
	err = cudaMemcpy(pairsGPU, chars_comparision, CHARS * CHARS, cudaMemcpyHostToDevice);
	validate(err);

// Copy weights array from host to the GPU memory
	err = cudaMemcpy(weightsGPU, weights, sizeof(double) * WEIGHTS_NUM,
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
			compare_scores_and_swap(&results_array[i], &tmp);
		}
#pragma omp critical
		{
			compare_scores_and_swap(&tmp, res);
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

