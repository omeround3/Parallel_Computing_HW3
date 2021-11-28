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

#include "mpi.h"

#include "sequence_alignment.h"

/* Strings group declaration */
char first_type_groups[CONSRV_SIZE][STR_LEN] = {"NDEQ", "MILV", "FYW", "NEQK", "QHRK", "HY", "STA", "NHQK", "MILF"};

char second_type_groups[SEMI_CONSRV_SIZE][STR_LEN] = {"SAG", "SGND", "NEQHRK", "HFY", "ATV", "STPA", "NDEQHK", "FVLIM", "CSA", "STNK", "SNDEQK"};

double max_val;
int weights[WEIGHTS_NUM];
/* table for storing characters comparision answers */
char chars_comparision[CHARS][CHARS];

int main(int argc, char *argv[]) {
	int i, num_of_sequences;
	int procceses_amount;	/* number of processes*/
	int process_rank;	/* process rank (process ID) */
	int job_share_size, job_remainder, job_offset;	/* process job share size & remainder */
	int m_share, m_remainder;	/* method share size & remainder */
	int omp_from, omp_to;
	int cuda_from, cuda_to;
	Alignment res_arr[THREADS];
	Alignment tmp;
	Payload tmp2;
	MPI_Status status; /* return status for receive */
	// omp_set_num_threads(THREADS);
	
	MPI_Init(&argc, &argv);	/* MPI Intialization */
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank); /* Get process rank */
	MPI_Comm_size(MPI_COMM_WORLD, &procceses_amount); /* Get number of processes */

	double start = MPI_Wtime();

	Payload *data;
	Alignment *res;

	/* Only the master process handles the input and data initalization */
	if (process_rank == 0) {
		build_table();
		get_data(data, res, &num_of_sequences);
		printf("Line 61\n");
	}
	/* Defining MPI_TYPEs */
	MPI_Datatype AlignmentMPIType;
	MPI_Datatype a_types[5] = { MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT };
	int a_block_len[5] = { 1, 1, 1, 1, 1 };
	MPI_Aint disp[5];
	disp[0] = (char*) &tmp.offset - (char*) &tmp;
	disp[1] = (char*) &tmp.char_idx - (char*) &tmp;
	disp[2] = (char*) &tmp.char_val - (char*) &tmp;
	disp[3] = (char*) &tmp.alignment_score - (char*) &tmp;
	disp[4] = (char*) &tmp.max_score - (char*) &tmp;
	MPI_Type_create_struct(5, a_block_len, disp, a_types, &AlignmentMPIType);
	MPI_Type_commit(&AlignmentMPIType);

	MPI_Datatype PayloadMPIType;
	MPI_Datatype d_types[4] = { MPI_CHAR, MPI_CHAR, MPI_INT, MPI_INT };
	int d_block_len[4] = { SEQ1_SIZE + 1, SEQ2_SIZE + 1, 1, 1 };
	disp[0] = (char*) &tmp2.seq1 - (char*) &tmp2;
	disp[1] = (char*) &tmp2.seq2 - (char*) &tmp2;
	disp[2] = (char*) &tmp2.len - (char*) &tmp2;
	disp[3] = (char*) &tmp2.max_offset - (char*) &tmp2;
	MPI_Type_create_struct(4, d_block_len, disp, d_types, &PayloadMPIType);
	MPI_Type_commit(&PayloadMPIType);

	if (process_rank == 0) {
		fprintf(stdout, "Process rank %d | Seq2: %s\n", process_rank, data[0].seq2);
	}
/* 	Broadcasting data for all MPI's processes */
	// MPI_Bcast(&data[0], 1, PayloadMPIType, MASTER_PROCESS, MPI_COMM_WORLD);
	// MPI_Bcast(&res, num_of_sequences, AlignmentMPIType, MASTER_PROCESS, MPI_COMM_WORLD);
	// MPI_Bcast(&max_val, 1, MPI_DOUBLE, MASTER_PROCESS, MPI_COMM_WORLD);
	// MPI_Bcast(&weights, WEIGHTS_NUM, MPI_LONG, MASTER_PROCESS, MPI_COMM_WORLD);
	// MPI_Bcast(&chars_comparision, CHARS * CHARS, MPI_CHAR, MASTER_PROCESS, MPI_COMM_WORLD);

	// for (int i = 0; i < num_of_sequences; i++)
	// {
		// printf("Process rank %d | Seq2: %s\n", process_rank, data[0].seq2);
	// }
	

// // Calculate total process share
// 	job_share_size = data.len / procceses_amount;
// 	job_remainder = data.len % procceses_amount;0
// 	job_offset = process_rank * job_share_size;

// // Calculate omp/CUDA (method) share
// 	m_share = job_share_size / 2;
// 	m_remainder = job_share_size % procceses_amount;

// // Set start/end points
// 	omp_from = job_offset;
// 	omp_to = job_offset + m_share;
// 	cuda_from = omp_to;
// 	if (process_rank != (procceses_amount - 1))
// 		cuda_to = job_offset + 2 * m_share + m_remainder;
// 	else
// 		cuda_to = job_offset + 2 * m_share + m_remainder + job_remainder;

// // Initialize res_arr
// 	for (i = 0; i < THREADS; ++i) {
// 		copy(&res, &res_arr[i]);
// 	}

// // CUDA function
// 	computeOnGPU(&data, &res, *chars_comparision, weights, cuda_from, cuda_to);

// // Each thread will replace from-to chars and will find optimal offset
// 	omp_set_num_threads(THREADS);
// #pragma omp parallel for private(i)
// 	for (i = omp_from; i < omp_to; i++) {
// 		find_mutants(&data, &res_arr[omp_get_thread_num()], i);
// 	}

// // Find optimal res in threads results
// 	for (i = 0; i < THREADS; ++i) {
// 		compare_and_swap(&res_arr[i], &res);
// 	}

// // Get results from processes, compare and swap if needed
// 	if (process_rank != 0) {
// 		MPI_Send(&res, 1, AlignmentMPIType, 0, 0, MPI_COMM_WORLD);
// 	} else {
// 		for (int process = 0; process < procceses_amount - 1; ++process) {
// 			MPI_Recv(&tmp, 1, AlignmentMPIType, MPI_ANY_SOURCE, 0,
// 			MPI_COMM_WORLD, &status);
// 			compare_and_swap(&tmp, &res);;
// 		}
// 		double end = MPI_Wtime();
// 		printf("Execution time is %f\n", end - start ); 
// 		output_print(&data, &res, output);
// 	}
// 	MPI_Finalize();

	return 0;
}

Alignment* find_mutants(const Payload *source, Alignment *res, int idx) {
	Alignment *tmp = (Alignment*) malloc(sizeof(Alignment));
	tmp->char_idx = idx;

	int c1 = source->seq2[idx] - 'A';
	for (int i = 1; i < CHARS && not_opt(res); ++i) {
		if (chars_comparision[c1][(c1 + i) % CHARS] != '%') {
			tmp->char_val = (c1 + i) % CHARS;
			find_offset(source, tmp);
			compare_and_swap(tmp, res);
		}
	}

	free(tmp);
	return res;
}

Alignment* find_offset(const Payload *source, Alignment *res) {

	Alignment *tmp = (Alignment*) malloc(sizeof(Alignment));
	tmp = copy(res, tmp);

	printf("Max offset: %d\n", source->max_offset);
	printf("Max Score: %d\n", res->max_score);
	for (int i = 0; i <= source->max_offset && not_opt(res); ++i) {
		printf("In find offset:: iteration %d\n", i);
		tmp->offset = i;
		// compare(source, tmp);
		// compare_and_swap(tmp, res);
	}
	printf("In find offeset:: EXIT LOOP\n");
	free(tmp);
	return res;

}

/* The function compares the alignment score and swaps the results if needed */
Alignment* compare_and_swap(const Alignment *a1, Alignment *a2) {
	if (a1->alignment_score > a2->alignment_score) {
		copy(a1, a2);
	}
	return a2;
}

int not_opt(Alignment *a) {
	// printf("Not Opt Value: %d\n", !(a->alignment_score == a->max_score));
	return !(a->alignment_score == a->max_score);
}

Alignment* copy(const Alignment *source, Alignment *dest) {
	dest->offset = source->offset;
	dest->char_idx = source->char_idx;
	dest->char_val = source->char_val;
	dest->alignment_score = source->alignment_score;
	dest->max_score = source->max_score;
	return dest;
}

/* 
The functions compares characters between the 2 sequences in the Payload,
and calculates the alignment score for a sequence #2
*/
Alignment* compare(const Payload *d, Alignment *a) {

	int c1, c2;
	a->alignment_score = 0;
	for (int chr_ofst = 0; chr_ofst < d->len; ++chr_ofst) {
		
		c1 = d->seq1[chr_ofst + a->offset] - 'A';	/* Convert char to ABC order index */
		if (chr_ofst == a->char_idx) {
			c2 = a->char_val;
		} else {
			c2 = d->seq2[chr_ofst] - 'A';
		}
		switch (chars_comparision[c1][c2]) {
		case '$':
			a->alignment_score += weights[0];
			break;
		case '%':
			a->alignment_score -= weights[1];
			break;
		case '#':
			a->alignment_score -= weights[2];
			break;
		default:
			a->alignment_score -= weights[3];
			break;
		}
	}
	return a;
}

void get_data(Payload *data, Alignment *res, int *num_of_sequences)
{
	char seq1[SEQ1_SIZE];

	/* Get input from stdin */
	fscanf(stdin, "%d %d %d %d", &weights[0], &weights[1], &weights[2],
		   &weights[3]);
	fscanf(stdin, "%s", seq1);
	fscanf(stdin, "%d", num_of_sequences);
	int seq_amount = *num_of_sequences;
	data = (Payload *)malloc(sizeof(Payload) * seq_amount);
	res = (Alignment *)malloc(sizeof(Alignment) * seq_amount);

	for (int i = 0; i < seq_amount; i++)
	{
		// data[i].seq1 = strdup(seq1);
		printf("Line 258\n");
		strcpy(data[i].seq1, seq1);
		fscanf(stdin, "%s", data[i].seq2);
		data[i].len = strlen(data[i].seq2);
		data[i].max_offset = strlen(seq1) - data[i].len;
		res[i].char_idx = -1;
		res[i].char_val = 0;
		res[i].offset = 0;
		res[i].max_score = strlen(data->seq2) * weights[0]; // stop if reached
		compare(&data[i], &res[i]);
		find_offset(&data[i], &res[i]);
		
		fprintf(stdout,"Data %d | Seq 2: %s\n",i,data[i].seq2);
		fprintf(stdout,"Data %d | AS: %d\n",i+1, res[i].alignment_score);
	}
	
}

/* 
The functions builds a characters table with the size CHARS * CHARS and fills it with signs
that corresponding to the rules of alignment score.
- Two identical characters are marked with the sign '$'
- Non-identical characters that belong to the first_type_groups strings are marked with the sign '%'
- Non-identical characters that belong to the second_type_groups strings are marked with the sign '#'
*/
void build_table() {
	for (int row = 0; row < CHARS; ++row) {
		for (int column = 0; column < CHARS; ++column) {
			chars_comparision[row][column] = (row == column) ? '$' : ' ';
		}
	}
	
	/* Fill first type groups characters */
	for (int i = 0; i < CONSRV_SIZE; ++i) {
		insert_string(first_type_groups[i], '%');
	}

	/* Fill second type groups characters */
	for (int i = 0; i < SEMI_CONSRV_SIZE; ++i) {
		insert_string(second_type_groups[i], '#');
	}
}

/* The function insert strings from groups into the characters table */
void insert_string(const char *str, const char sign) {
	int c1, c2;
	for (int chr_idx = 0; chr_idx < strlen(str); ++chr_idx) {
		c1 = str[chr_idx] - 'A';
		for (int offset = chr_idx + 1; offset < strlen(str); ++offset) {
			c2 = str[offset] - 'A';
			if (chars_comparision[c1][c2] == ' ') {
				chars_comparision[c1][c2] = sign;
				chars_comparision[c2][c1] = sign;
			}
		}
	}
}

void output_print(Payload *data, Alignment *res) {
	char str[SEQ2_SIZE + 1];
	char* hyphen_pos;
	strcpy(str, data->seq2);
	if (res->char_idx != -1)
		str[res->char_idx] = 'A' + res->char_val;
	fprintf(stdout, "%s\n",str);
	fprintf(stdout,"%d %d",res->offset,res->alignment_score);
}
