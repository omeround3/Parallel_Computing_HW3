#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <string.h>
#include "sequence_alignment.h"


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
__device__ void dev_compare(const Payload *payload, Score *score, char *chars_comparision,
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

__device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void find_optimal_offset_cuda(Payload *source, Score *score, char *chars_comparision, int *weights, int offset)
{
	Score tmp;
	int tid = threadIdx.x;
	int col_offset = blockDim.x * blockIdx.x;
	int gid = tid + col_offset;

	/* Run first compare on source score and then compare to tmp scores */
	// dev_compare(source, score, chars_comparision, weights);
	dev_deep_copy_score(score, &tmp);
	
	/* Each block will be responsible for an offset */
	tmp.offset = offset;

	/* Each thread in a block will calculate score for each hyphen (seq2 length threads in total) */
	tmp.hyphen_idx = gid;
	__syncthreads();
	if (gid < source->len)
	{
		dev_compare(source, &tmp, chars_comparision, weights);
		dev_compare_scores_and_swap(&tmp, score);
		/* hyphen_idx = 0 -> means the hyphen is at the end of the string (e.g. ABC-) */
		if (score->hyphen_idx == 0)
		{
			score->hyphen_idx = source->len;
		}
	}
}

void check_for_error(cudaError_t err)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void cuda_free(void *ptr, cudaError_t err)
{
	// Free allocated memory on GPU - ArrayA
	if (cudaFree(ptr) != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device data - %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

Score *cuda_calculation(Payload *source, Score *score, char *chars_comparision, int *weights, int start_offset, int end_offset)
{

	Score tmp;
	deep_copy_score(score, &tmp);

	cudaDeviceProp dev_properties;
	cudaGetDeviceProperties(&dev_properties, 0);
	/* The number of blocks is equal to the offset range given to CUDA to calculate */
	// int num_of_blocks = end_offset - start_offset;
	int num_of_blocks = source->len / dev_properties.maxThreadsPerBlock + 1;
	int num_of_threads = dev_properties.maxThreadsPerBlock;
	int offset = end_offset - start_offset;
	/* Error code for verifying CUDA function */
	cudaError_t err = cudaSuccess;

	/* Device (GPU) memory allocation for source */
	Payload *dev_source;
	err = cudaMalloc((void **)&dev_source, sizeof(Payload));
	check_for_error(err);

	/* Device (GPU) memory allocation for score */
	Score *dev_score;
	err = cudaMalloc((void **)&dev_score, sizeof(Score));
	check_for_error(err);

	/* Device (GPU) memory allocation for chars_comparision */
	char *dev_chars_comparision;
	err = cudaMalloc((void **)&dev_chars_comparision, CHARS * CHARS);
	check_for_error(err);

	/* Device (GPU) memory allocation for weights */
	int *dev_weights;
	err = cudaMalloc((void **)&dev_weights, sizeof(int) * WEIGHTS_NUM);
	check_for_error(err);

	/* Memory copy from host to device (GPU) */
	err = cudaMemcpy(dev_source, source, sizeof(Payload), cudaMemcpyHostToDevice);
	check_for_error(err);

	/* Memory copy from host to device (GPU) */
	err = cudaMemcpy(dev_score, score, sizeof(Score), cudaMemcpyHostToDevice);
	check_for_error(err);

	/* Memory copy from host to device (GPU) */
	err = cudaMemcpy(dev_chars_comparision, chars_comparision, CHARS * CHARS, cudaMemcpyHostToDevice);
	check_for_error(err);

	/* Memory copy from host to device (GPU) */
	err = cudaMemcpy(dev_weights, weights, sizeof(int) * WEIGHTS_NUM,
					 cudaMemcpyHostToDevice);
	check_for_error(err);

	for (int i = 0; i < offset; ++i)
	{
		find_optimal_offset_cuda<<<num_of_blocks, num_of_threads>>>(dev_source, dev_score, dev_chars_comparision, dev_weights, i);
		err = cudaGetLastError();
		check_for_error(err);
	}
	// Launch the Kernel
	

	// Copy the  result from GPU to the host memory.
	err = cudaMemcpy(score, dev_score, sizeof(Score), cudaMemcpyDeviceToHost);
	check_for_error(err);

	// Free allocated space in GPU
	cuda_free(dev_source, err);
	cuda_free(dev_score, err);
	cuda_free(dev_chars_comparision, err);
	cuda_free(dev_weights, err);
	return score;
}
