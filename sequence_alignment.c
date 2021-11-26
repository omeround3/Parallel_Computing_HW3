/* 
	This program will calculate the Sequence Alignment scores of sequences in the input (input file). The calculation proceess involves MPI, OpenMP and CUDA.

	Sequence Alignment – Estimatation of similarity of two strings of letters 
	Sequence - a string of capital letters
	Alignment Score - The score given to comparision between two sequences. The higher the value, the more similar the sequences.

	Input: 
	Output: 
	
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
double weights[WEIGHTS];
/* table for storing characters comparision answers */
char chars_comparision[CHARS][CHARS];

int main(int argc, char *argv[]) {
	omp_set_num_threads(THREADS);
	char input[] = "input.txt";
	char output[] = "output.txt";
	int i;
	int p; /*number of processes*/
	int my_rank; /* rank of process */
	int p_share, p_remainder, offset; /* process share size & remainder */
	int m_share, m_remainder;/* method share size & remainder */
	int omp_from, omp_to;
	int cuda_from, cuda_to;
	Alignment res_arr[THREADS];
	Alignment tmp;
	MPI_Status status; /* return status for receive */

	/* start up MPI */
	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	double start = MPI_Wtime();

// Set Data struct instance
	Data data;
// Set Result struct instance
	Alignment res;

// Master process reads from file
	if (my_rank == 0) {
		init_matrix(); // fill matrix according to given rules
		load_from_file(&data, &res, input);
	}
// Defining MPI_TYPEs
	MPI_Datatype AlignmentMPIType;
	MPI_Datatype a_types[4] = { MPI_INT, MPI_INT, MPI_INT, MPI_DOUBLE };
	int a_block_len[4] = { 1, 1, 1, 1 };
	MPI_Aint disp[4];
	disp[0] = (char*) &res.offset - (char*) &res;
	disp[1] = (char*) &res.char_idx - (char*) &res;
	disp[2] = (char*) &res.char_val - (char*) &res;
	disp[3] = (char*) &res.score - (char*) &res;
	MPI_Type_create_struct(4, a_block_len, disp, a_types, &AlignmentMPIType);
	MPI_Type_commit(&AlignmentMPIType);

	MPI_Datatype DataMPIType;
	MPI_Datatype d_types[4] = { MPI_CHAR, MPI_CHAR, MPI_INT, MPI_INT };
	int d_block_len[4] = { SQ1_SIZE + 1, SQ2_SIZE + 1, 1, 1 };
	disp[0] = (char*) &data.seq1 - (char*) &data;
	disp[1] = (char*) &data.seq2 - (char*) &data;
	disp[2] = (char*) &data.len - (char*) &data;
	disp[3] = (char*) &data.max_offset - (char*) &data;
	MPI_Type_create_struct(4, d_block_len, disp, d_types, &DataMPIType);
	MPI_Type_commit(&DataMPIType);

// Broadcasting
	MPI_Bcast(&data, 1, DataMPIType, 0, MPI_COMM_WORLD);
	MPI_Bcast(&res, 1, AlignmentMPIType, 0, MPI_COMM_WORLD);
	MPI_Bcast(&max_val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&weights, WEIGHTS, MPI_LONG, 0, MPI_COMM_WORLD);
	MPI_Bcast(&chars_comparision, CHARS * CHARS, MPI_CHAR, 0, MPI_COMM_WORLD);

// Calculate total process share
	p_share = data.len / p;
	p_remainder = data.len % p;
	offset = my_rank * p_share;

// Calculate omp/CUDA (method) share
	m_share = p_share / 2;
	m_remainder = p_share % p;

// Set start/end points
	omp_from = offset;
	omp_to = offset + m_share;
	cuda_from = omp_to;
	if (my_rank != (p - 1))
		cuda_to = offset + 2 * m_share + m_remainder;
	else
		cuda_to = offset + 2 * m_share + m_remainder + p_remainder;

// Initialize res_arr
	for (i = 0; i < THREADS; ++i) {
		copy(&res, &res_arr[i]);
	}

// CUDA function
	computeOnGPU(&data, &res, *chars_comparision, weights, cuda_from, cuda_to);

// Each thread will replace from-to chars and will find optimal offset
	omp_set_num_threads(THREADS);
#pragma omp parallel for private(i)
	for (i = omp_from; i < omp_to; i++) {
		find_mutants(&data, &res_arr[omp_get_thread_num()], i);
	}

// Find optimal res in threads results
	for (i = 0; i < THREADS; ++i) {
		compare_and_swap(&res_arr[i], &res);
	}

// Get results from processes, compare and swap if needed
	if (my_rank != 0) {
		MPI_Send(&res, 1, AlignmentMPIType, 0, 0, MPI_COMM_WORLD);
	} else {
		for (int process = 0; process < p - 1; ++process) {
			MPI_Recv(&tmp, 1, AlignmentMPIType, MPI_ANY_SOURCE, 0,
			MPI_COMM_WORLD, &status);
			compare_and_swap(&tmp, &res);;
		}
		double end = MPI_Wtime();
		printf("Execution time is %f\n", end - start ); 
		print_to_file(&data, &res, output);
	}
	MPI_Finalize();

	return 0;
}

Alignment* find_mutants(const Data *source, Alignment *res, int idx) {
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

Alignment* find_offset(const Data *source, Alignment *res) {

	Alignment *tmp = (Alignment*) malloc(sizeof(Alignment));
	tmp = copy(res, tmp);

	for (int i = 0; i <= source->max_offset && not_opt(res); ++i) {
		tmp->offset = i;
		compare(source, tmp);
		compare_and_swap(tmp, res);
	}
	free(tmp);
	return res;

}

Alignment* compare_and_swap(const Alignment *a1, Alignment *a2) {
	if (a1->score > a2->score) {
		copy(a1, a2);
	}
	return a2;
}

int not_opt(Alignment *a) {
	return !(a->score == max_val);
}

Alignment* copy(const Alignment *source, Alignment *dest) {
	dest->offset = source->offset;
	dest->score = source->score;
	dest->char_idx = source->char_idx;
	dest->char_val = source->char_val;
	return dest;
}

Alignment* compare(const Data *d, Alignment *a) {
	int c1, c2;
	a->score = 0;
	for (int chr_ofst = 0; chr_ofst < d->len; ++chr_ofst) {
		c1 = d->seq1[chr_ofst + a->offset] - 'A';
		if (chr_ofst == a->char_idx) {
			c2 = a->char_val;
		} else {
			c2 = d->seq2[chr_ofst] - 'A';
		}
		switch (chars_comparision[c1][c2]) {
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

Data* load_from_file(Data *data, Alignment *res, char *file) {
	FILE *fp;
	char buff[8];
	char *hyphen_pos;

// Read from file
	fp = fopen(file, "r");
	fscanf(fp, "%lf %lf %lf %lf", &weights[0], &weights[1], &weights[2],
			&weights[3]);
	fscanf(fp, "%s", data->seq1);
	fscanf(fp, "%s", data->seq2);
	fscanf(fp, "%s", buff);

// Replace all hyphens seq1
	hyphen_pos = strchr(data->seq1, '-');
	while (hyphen_pos) {
		*hyphen_pos = ('Z' + 1);
		hyphen_pos = strchr(hyphen_pos, '-');
	}
// Replace all hyphens seq2
	hyphen_pos = strchr(data->seq2, '-');
	while (hyphen_pos) {
		*hyphen_pos = ('Z' + 1);
		hyphen_pos = strchr(hyphen_pos, '-');
	}

	max_val = strlen(data->seq2) * weights[0]; // stop if reached

// Initialize
	data->max_offset = strlen(data->seq1) - strlen(data->seq2);
	data->len = strlen(data->seq2);
	res->char_idx = -1;
	res->char_val = 0;
	res->offset = 0;
	compare(data, res);
	find_offset(data, res);
	fclose(fp);
	return data;
}
void init_matrix() {
	for (int row = 0; row < CHARS; ++row) {
		for (int column = 0; column < CHARS; ++column) {
			chars_comparision[row][column] = (row == column) ? '$' : ' ';
		}
	}

// Fill conservative groups chars
	for (int str_idx = 0; str_idx < CONSRV_SIZE; ++str_idx) {
		insert_str_to_matrix(first_type_groups[str_idx], '%');
	}

// Fill semi-conservative groups chars
	for (int str_idx = 0; str_idx < SEMI_CONSRV_SIZE; ++str_idx) {
		insert_str_to_matrix(second_type_groups[str_idx], '#');
	}
}

// Process groups int a matrix
void insert_str_to_matrix(const char *str, const char sign) {
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

void print_to_file(Data *data, Alignment *res, char *path) {
	FILE *fp = fopen(path, "w+");
	char str[SQ2_SIZE + 1];
	char* hyphen_pos;
	strcpy(str, data->seq2);
	if (res->char_idx != -1)
		str[res->char_idx] = 'A' + res->char_val;
// Place back hyphens
	hyphen_pos = strchr(str, ('Z'+1));
	while (hyphen_pos) {
		*hyphen_pos = '-';
		hyphen_pos = strchr(hyphen_pos, ('Z'+1));
	}
	fprintf(fp, "%s\n",str);
	fprintf(fp,"%d %lf",res->offset,res->score);
	fclose(fp);
}
