/*
	This program will calculate the Sequence Alignment scores of sequences in the input (input file). The calculation proceess involves MPI, OpenMP and CUDA.

	Sequence Alignment – Estimatation of similarity of two strings of letters
	Sequence - a string of capital letters
	Alignment Score - The score given to comparision between two sequences. The higher the value, the more similar the sequences.

	Input: letters sequences for comparision
	Output: For each sequence, the offset (n) and hypen location in the mutant sequence (k) which procudes maxial alignment score when comparing to the first sequence

	Author:
	Omer Lev-Ron
*/
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "mpi.h"

#include "sequence_alignment.h"

/* Strings group declaration */
char first_type_groups[FIRST_TYPE_GROUP_SIZE][GROUPS_STRINGS_SIZE] = {"NDEQ", "MILV", "FYW", "NEQK", "QHRK", "HY", "STA", "NHQK", "MILF"};

char second_type_groups[SECOND_TYPE_GROUP_SIZE][GROUPS_STRINGS_SIZE] = {"SAG", "SGND", "NEQHRK", "HFY", "ATV", "STPA", "NDEQHK", "FVLIM", "CSA", "STNK", "SNDEQK"};

int weights[WEIGHTS_NUM];			  /* array to hold weights values */
char chars_comparision[CHARS][CHARS]; /* table for storing characters comparision answers */

int main(int argc, char *argv[])
{
	int num_of_sequences;
	int procceses_amount; /* number of processes*/
	int process_rank;	  /* process rank (process ID) */
	int work_size;
	int offset_remainder;
	int cuda_start_offset, cuda_end_offset;
	int cuda_omp_work_size, cuda_omp_offset_reminder;
	int omp_start_offset, omp_end_offset;

	MPI_Status status; /* return status for receive */

	/* Initialize MPI */
	MPI_Init(&argc, &argv);							  /* send MPI command line arguments  */
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);	  /* Get process rank */
	MPI_Comm_size(MPI_COMM_WORLD, &procceses_amount); /* Get number of processes */
	double start, end;
	start = MPI_Wtime();

	Payload *payload;
	Score *scores;

	/* Check if number of processes is 2 or more */
	if (procceses_amount < 2)
	{
		fprintf(stderr, "%s: The program requires at least two processors. Use -n 2 or bigger.", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
	}

	/* Only the master process handles the input and data initalization */
	if (process_rank == 0)
	{
		build_table();

		/* Get input from stdin */
		char seq1[SEQ1_SIZE];
		fscanf(stdin, "%d %d %d %d", &weights[0], &weights[1], &weights[2],
			   &weights[3]);
		/* Get sequence 1 and convert to uppercase */
		fscanf(stdin, "%s", seq1);
		for (int i = 0; i < strlen(seq1); i++)
		{
			if (seq1[i] >= 'a' && seq1[i] <= 'z')
				seq1[i] = toupper(seq1[i]);
		}

		/* Allocate memory for structs */
		fscanf(stdin, "%d", &num_of_sequences);
		payload = (Payload *)malloc(sizeof(Payload) * num_of_sequences);
		scores = (Score *)malloc(sizeof(Score) * num_of_sequences);

		/* Get the sequeneces for comparision and save into structs */
		for (int i = 0; i < num_of_sequences; i++)
		{
			strcpy(payload[i].seq1, seq1);
			fscanf(stdin, "%s", payload[i].seq2);
			payload[i].len = strlen(payload[i].seq2);
			for (int j = 0; j < payload[j].len; j++)
			{
				if (payload[i].seq2[j] >= 'a' && payload[i].seq2[j] <= 'z')
					payload[i].seq2[j] = toupper(payload[i].seq2[j]);
			}
			payload[i].max_offset = strlen(seq1) - payload[i].len;
			scores[i].hyphen_idx = 0;
			scores[i].offset = 0;
			scores[i].alignment_score = 0;
			scores[i].max_score = strlen(payload[i].seq2) * weights[0];
		}
	}

	/* Defining MPI_TYPEs */
	Score tmp;
	MPI_Datatype ScoreMPIType;
	MPI_Datatype score_types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT};
	int score_block_len[4] = {1, 1, 1, 1};
	MPI_Aint disp1[4];
	disp1[0] = (char *)&tmp.offset - (char *)&tmp;
	disp1[1] = (char *)&tmp.hyphen_idx - (char *)&tmp;
	disp1[3] = (char *)&tmp.alignment_score - (char *)&tmp;
	disp1[4] = (char *)&tmp.max_score - (char *)&tmp;
	MPI_Type_create_struct(4, score_block_len, disp1, score_types, &ScoreMPIType);
	MPI_Type_commit(&ScoreMPIType);

	Payload tmp2;
	MPI_Datatype PayloadMPIType;
	MPI_Datatype paylod_types[4] = {MPI_CHAR, MPI_CHAR, MPI_INT, MPI_INT};
	int paylod_block_len[4] = {SEQ1_SIZE + 1, SEQ2_SIZE + 1, 1, 1};
	MPI_Aint disp2[4];
	disp2[0] = (char *)&tmp2.seq1 - (char *)&tmp2;
	disp2[1] = (char *)&tmp2.seq2 - (char *)&tmp2;
	disp2[2] = (char *)&tmp2.len - (char *)&tmp2;
	disp2[3] = (char *)&tmp2.max_offset - (char *)&tmp2;
	MPI_Type_create_struct(4, paylod_block_len, disp2, paylod_types, &PayloadMPIType);
	MPI_Type_commit(&PayloadMPIType);

	/* 	Broadcasting shared values for all MPI's processes */
	MPI_Bcast(&num_of_sequences, 1, MPI_INT, MASTER_PROCESS, MPI_COMM_WORLD);
	MPI_Bcast(&weights, WEIGHTS_NUM, MPI_INT, MASTER_PROCESS, MPI_COMM_WORLD);
	MPI_Bcast(&chars_comparision, CHARS * CHARS, MPI_CHAR, MASTER_PROCESS, MPI_COMM_WORLD);
	/* Allocation for other processes */
	if (process_rank != 0)
	{
		payload = (Payload *)malloc(sizeof(Payload) * num_of_sequences);
		scores = (Score *)malloc(sizeof(Score) * num_of_sequences);
	}
	/* 	Broadcasting shared structs arrays for all MPI's processes */
	MPI_Bcast(payload, num_of_sequences, PayloadMPIType, MASTER_PROCESS, MPI_COMM_WORLD);
	MPI_Bcast(scores, num_of_sequences, ScoreMPIType, MASTER_PROCESS, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	/* Calculate score for each sequence, each processes takes a part of the max_offset. */
	if (process_rank == 0)
	{
		Score tmp_score;
		for (int i = 0; i < num_of_sequences; ++i)
		{
			/* Calculate offset for each sequence per process */
			work_size = payload[i].max_offset / procceses_amount;
			offset_remainder = payload[i].max_offset % procceses_amount;

			/* Calculate OpenMP/CUDA work size' */
			cuda_omp_work_size = work_size / 2;
			cuda_omp_offset_reminder = work_size % procceses_amount;

			/* Set start/end offsets */
			cuda_start_offset = process_rank * work_size;
			cuda_end_offset = cuda_start_offset + cuda_omp_work_size;
			omp_start_offset = cuda_end_offset;
			omp_end_offset = omp_start_offset + cuda_omp_work_size + cuda_omp_offset_reminder;

			/* Master process to do his work size */
			/* Send to CUDA and OpenMP their offset part */
			cuda_calculation(&payload[i], &scores[i], *chars_comparision, weights, cuda_start_offset, cuda_end_offset);
			find_optimal_offset_omp(&payload[i], &scores[i], omp_start_offset, omp_end_offset);

			/* Recieve results from other processes and   */
			for (int j = 1; j < procceses_amount; j++)
			{
				MPI_Recv(&tmp_score, 1, ScoreMPIType, j, RESULT_TAG, MPI_COMM_WORLD, &status);
				compare_scores_and_swap(&tmp_score, &scores[i]);
			}
		}
	}
	else
	{
		for (int i = 0; i < num_of_sequences; i++)
		{
			/* Calculate offset for each sequence per process */
			work_size = payload[i].max_offset / procceses_amount;
			offset_remainder = payload[i].max_offset % procceses_amount;

			/* Calculate OpenMP/CUDA work size' */
			cuda_omp_work_size = work_size / 2;
			cuda_omp_offset_reminder = work_size % procceses_amount;

			/* Set start/end offsets */
			cuda_start_offset = process_rank * work_size;
			cuda_end_offset = cuda_start_offset + cuda_omp_work_size;
			omp_start_offset = cuda_end_offset;
			omp_end_offset = omp_start_offset + cuda_omp_work_size + cuda_omp_offset_reminder;

			/* Last process handles offset reminder if exists */
			if (process_rank == procceses_amount - 1 && offset_remainder != 0)
			{
				omp_end_offset += offset_remainder;
			}
			
			/* Send to CUDA and OpenMP their offset part */
			cuda_calculation(&payload[i], &scores[i], *chars_comparision, weights, cuda_start_offset, cuda_end_offset);
			find_optimal_offset_omp(&payload[i], &scores[i], omp_start_offset, omp_end_offset);

			/* Send results to Master Process */
			MPI_Send(&scores[i], 1, ScoreMPIType, MASTER_PROCESS, RESULT_TAG, MPI_COMM_WORLD);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/* Print Results */
	if (process_rank == 0)
	{
		end = MPI_Wtime();
		printf("Execution time is %f\n", end - start);
		results_output(scores, num_of_sequences);
	}

	free(payload);
	free(scores);

	MPI_Finalize();
	return 0;
}

/* 
This function finds the optimal offset(n) and mutant (k) for the 2 given sequences 
in the Payload. The start_offset and end_offset are the part the functions gets for
its calculations.   

Inner loop uses OpenMP for mutant(k) calculations
*/
Score *find_optimal_offset_omp(const Payload *source, Score *score, int start_offset, int end_offset)
{
	Score *tmp = (Score *)malloc(sizeof(Score));


	/* Run first calculate_score on source score and then compare to tmp score */
	calculate_score(source, score);
	tmp = deep_copy_score(score, tmp);

	for (int i = start_offset; i <= end_offset - 1; ++i)
	{
		tmp->offset = i;
		/* for each hyphen offset find optimal */
		#pragma omp parallel for default(none) firstprivate(tmp) shared(source, score, start_offset, end_offset)
		for (int j = 1; j < source->len; ++j)
		{
			tmp->hyphen_idx = j;
			#pragma omp critical
			{
				calculate_score(source, tmp);
				compare_scores_and_swap(tmp, score);
			}
		}
	}
	/* hyphen_idx = 0 -> means the hyphen is at the end of the string (e.g. ABC-) */
	if (score->hyphen_idx == 0)
	{
		score->hyphen_idx = source->len;
	}

	free(tmp);
	return score;
}

/* 
This function finds the optimal offset(n) and mutant (k) for the 2 given sequences 
in the Payload. The start_offset and end_offset are the part the functions gets for
its calculations.   

This function was used for MPI calculations without OpenMP or CUDA
*/
Score *find_optimal_offset(const Payload *source, Score *score, int start_offset, int end_offset)
{
	Score *tmp = (Score *)malloc(sizeof(Score));

	/* Run first calculate_score on source score and then compare to tmp score */
	calculate_score(source, score);
	tmp = deep_copy_score(score, tmp);

	for (int i = start_offset; i <= end_offset - 1 && is_score_optimized(score); ++i)
	{
		tmp->offset = i;

		/* for each hyphen offset find optimal */
		for (int j = 1; j < source->len && is_score_optimized(score); ++j)
		{
			tmp->hyphen_idx = j;
			calculate_score(source, tmp);
			compare_scores_and_swap(tmp, score);
		}
	}
	/* hyphen_idx = 0 -> means the hyphen is at the end of the string (e.g. ABC-) */
	if (score->hyphen_idx == 0)
	{
		score->hyphen_idx = source->len;
	}

	free(tmp);
	return score;
}

/* The function compares the alignment score and swaps the results if needed */
void compare_scores_and_swap(const Score *s1, Score *s2)
{
	if (s1->alignment_score > s2->alignment_score)
	{
		deep_copy_score(s1, s2);
	}
}

/* An helper method to optimize calculations; the method checks if current score is the max */
int is_score_optimized(Score *score)
{
	return !(score->alignment_score == score->max_score);
}

/* Deep copy of one Score instance to another */
Score *deep_copy_score(const Score *source, Score *dest)
{
	dest->offset = source->offset;
	dest->hyphen_idx = source->hyphen_idx;
	dest->alignment_score = source->alignment_score;
	dest->max_score = source->max_score;
	return dest;
}

/*
The functions compares characters between the 2 sequences in the Payload,
and calculates the alignment score for a sequence #2
*/
void calculate_score(const Payload *payload, Score *score)
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
		switch (chars_comparision[seq1_char][seq2_char])
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

/*
The functions builds a characters table with the size CHARS * CHARS and fills it with signs
that corresponding to the rules of alignment score.
- Two identical characters are marked with the sign '$'
- Non-identical characters that belong to the first_type_groups strings are marked with the sign '%'
- Non-identical characters that belong to the second_type_groups strings are marked with the sign '#'
*/
void build_table()
{
	for (int row = 0; row < CHARS; ++row)
	{
		for (int column = 0; column < CHARS; ++column)
		{
			chars_comparision[row][column] = (row == column) ? '$' : ' ';
		}
	}

	/* Fill first type groups characters */
	for (int i = 0; i < FIRST_TYPE_GROUP_SIZE; ++i)
	{
		insert_string(first_type_groups[i], '%');
	}

	/* Fill second type groups characters */
	for (int i = 0; i < SECOND_TYPE_GROUP_SIZE; ++i)
	{
		insert_string(second_type_groups[i], '#');
	}
}

/* The function insert strings from groups into the characters table */
void insert_string(const char *str, const char sign)
{
	int c1, c2;
	for (int chr_idx = 0; chr_idx < strlen(str); ++chr_idx)
	{
		c1 = str[chr_idx] - 'A';
		for (int offset = chr_idx + 1; offset < strlen(str); ++offset)
		{
			c2 = str[offset] - 'A';
			if (chars_comparision[c1][c2] == ' ')
			{
				chars_comparision[c1][c2] = sign;
				chars_comparision[c2][c1] = sign;
			}
		}
	}
}

/* This functions prints the calculations results */
void results_output(Score *scores, int num_sequences)
{
	fprintf(stdout, "The number of sequences in the input is: %d\n", num_sequences);
	for (int i = 0; i < num_sequences; ++i)
	{
		fprintf(stdout, "Sequence #%d | optimal offset(n): %d | optimal hyphen index(k): %d | alignment score: %d\n", i + 1, scores[i].offset, scores[i].hyphen_idx, scores[i].alignment_score);
	}
}