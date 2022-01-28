#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <string.h>
#include "sequence_alignment.h"

/* Deep copy of one Score instance to another */
__device__ Score *dev_deep_copy_score(const Score *source, Score *dest)
{
	dest->offset = source->offset;
	dest->hyphen_idx = source->hyphen_idx;
	dest->alignment_score = source->alignment_score;
	dest->max_score = source->max_score;
	return dest;
}

/* The function compares the alignment score and swaps the results if needed */
__device__ void dev_compare_scores_and_swap(const Score *s1, Score *s2)
{
	if (s1->alignment_score > s2->alignment_score)
	{
		dev_deep_copy_score(s1, s2);
	}
}

/*
The functions compares characters between the 2 sequences in the Payload,
and calculates the alignment score for a sequence #2
*/
__device__ void dev_calculate_score(const Payload *payload, Score *score, char *chars_comparision,
							  int *weights)
{
	int passed_hypen_flag = 0;
	int seq1_char, seq2_char;
	score->alignment_score = 0;
	for (int char_index = 0; char_index < payload->len; ++char_index)
	{
		/* Convert char to ABC order index */
		seq2_char = payload->seq2[char_index] - 'A';
		if ((char_index == score->hyphen_idx && score->hyphen_idx != 0) || passed_hypen_flag)
		{
			seq1_char = payload->seq1[char_index + 1 + score->offset] - 'A'; /* skip character if index is hyphen */
			passed_hypen_flag = 1;
		}
		else
		{
			seq1_char = payload->seq1[char_index + score->offset] - 'A';
		}
		/* check sign of characters */
		switch (chars_comparision[seq1_char * CHARS + seq2_char])
		{
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


__global__ void find_optimal_offset_cuda(Payload *source, Score *score, char *chars_comparision, int *weights, int offset)
{
	Score tmp;
	int tid = threadIdx.x;
	int col_offset = blockDim.x * blockIdx.x;
	int gid = tid + col_offset;

	/* Run first calculate_score on source score and then compare to tmp scores */
	// dev_calculate_score(source, score, chars_comparision, weights);
	dev_deep_copy_score(score, &tmp);
	
	/* Each block will be responsible for an offset */
	tmp.offset = offset;

	/* Each thread in a block will calculate score for each hyphen (seq2 length threads in total) */
	tmp.hyphen_idx = gid;
	__syncthreads();
	if (gid < source->len)
	{
		dev_calculate_score(source, &tmp, chars_comparision, weights);
		dev_compare_scores_and_swap(&tmp, score);
		/* hyphen_idx = 0 -> means the hyphen is at the end of the string (e.g. ABC-) */
		if (score->hyphen_idx == 0)
		{
			score->hyphen_idx = source->len;
		}
	}
}

void check_for_error(cudaError_t status_code)
{
	if (status_code != cudaSuccess)
	{
		fprintf(stderr, "%s\n", cudaGetErrorString(status_code));
		exit(EXIT_FAILURE);
	}
}

void dev_free(void *p, cudaError_t status_code)
{
	/* Free allocated memory on GPU  */
	if (cudaFree(p) != cudaSuccess)
	{
		fprintf(stderr, "Couldn't free device memory: %s\n", cudaGetErrorString(status_code));
		exit(EXIT_FAILURE);
	}
}

Score *cuda_calculation(Payload *source, Score *score, char *chars_comparision, int *weights, int start_offset, int end_offset)
{

	Score tmp;
	deep_copy_score(score, &tmp);

	cudaDeviceProp dev_properties;
	cudaGetDeviceProperties(&dev_properties, 0);
	/* The number of blocks depends on the length of the sequence to be compared to */
	int num_of_blocks = source->len / dev_properties.maxThreadsPerBlock + 1;
	/* Using maximum allowed threads per block */
	int num_of_threads = dev_properties.maxThreadsPerBlock;
	int offset = end_offset - start_offset;
	/* Error code for verifying CUDA functions */
	cudaError_t status_code = cudaSuccess;

	/* Device (GPU) memory allocation for source */
	Payload *dev_source;
	status_code = cudaMalloc((void **)&dev_source, sizeof(Payload));
	check_for_error(status_code);

	/* Device (GPU) memory allocation for chars_comparision */
	char *dev_chars_comparision;
	status_code = cudaMalloc((void **)&dev_chars_comparision, CHARS * CHARS);
	check_for_error(status_code);

	/* Device (GPU) memory allocation for score */
	Score *dev_score;
	status_code = cudaMalloc((void **)&dev_score, sizeof(Score));
	check_for_error(status_code);

	/* Memory copy from host to device (GPU) */
	status_code = cudaMemcpy(dev_source, source, sizeof(Payload), cudaMemcpyHostToDevice);
	check_for_error(status_code);

	/* Device (GPU) memory allocation for weights */
	int *dev_weights;
	status_code = cudaMalloc((void **)&dev_weights, sizeof(int) * WEIGHTS_NUM);
	check_for_error(status_code);

	/* Memory copy from host to device (GPU) */
	status_code = cudaMemcpy(dev_score, score, sizeof(Score), cudaMemcpyHostToDevice);
	check_for_error(status_code);

	/* Memory copy from host to device (GPU) */
	status_code = cudaMemcpy(dev_chars_comparision, chars_comparision, CHARS * CHARS, cudaMemcpyHostToDevice);
	check_for_error(status_code);

	/* Memory copy from host to device (GPU) */
	status_code = cudaMemcpy(dev_weights, weights, sizeof(int) * WEIGHTS_NUM,
					 cudaMemcpyHostToDevice);
	check_for_error(status_code);

	for (int i = 0; i < offset; ++i)
	{
		/* Kernel launch */
		find_optimal_offset_cuda<<<num_of_blocks, num_of_threads>>>(dev_source, dev_score, dev_chars_comparision, dev_weights, i);
		status_code = cudaGetLastError();
		check_for_error(status_code);
	}
	

	/* Send calculation result to Host from device (GPU) */
	status_code = cudaMemcpy(score, dev_score, sizeof(Score), cudaMemcpyDeviceToHost);
	check_for_error(status_code);

	/* Device (GPU) free allocation */
	dev_free(dev_source, status_code);
	dev_free(dev_score, status_code);
	dev_free(dev_chars_comparision, status_code);
	dev_free(dev_weights, status_code);

	return score;
}
