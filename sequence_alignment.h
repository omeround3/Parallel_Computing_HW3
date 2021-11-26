/*

*/

#ifndef SEQUENCE_ALIGNMENT_H_
#define SEQUENCE_ALIGNMENT_H_

#define WEIGHTS 4
#define STR_LEN 10
#define CHARS 27
#define CONSRV_SIZE 9
#define SEMI_CONSRV_SIZE 11
#define SQ1_SIZE 10000
#define SQ2_SIZE 5000
#define MY_MIN 0
#define MY_MAX 1
#define THREADS 2

#include <omp.h>

typedef struct Data {
	char seq1[SQ1_SIZE + 1];
	char seq2[SQ2_SIZE + 1];
	int len;
	int max_offset;
} Data;

typedef struct Alignments {
	int offset;
	int char_idx;
	int char_val;
	double score;
} Alignment;

int not_opt(Alignment *a);

void init_matrix();
void insert_str_to_matrix(const char *str, const char sign);
void print_to_file(Data *data, Alignment *res, char *path);

Alignment* find_mutants(const Data *source, Alignment *res, int idx);
Alignment* find_offset(const Data *source, Alignment *res);
Alignment* compare_and_swap(const Alignment *a1, Alignment *a2);
Alignment* copy(const Alignment *source, Alignment *dest);
Alignment* compare(const Data *d, Alignment *a);

Data* load_from_file(Data *data, Alignment *res, char *file);

Alignment* computeOnGPU(Data *source, Alignment *result, char *chars_comparision,
		double *weights, int from, int to);
#endif /* SEQUENCE_ALIGNMENT_H_ */
