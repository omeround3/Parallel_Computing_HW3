/*

*/

#ifndef SEQUENCE_ALIGNMENT_H_
#define SEQUENCE_ALIGNMENT_H_

#define WEIGHTS_NUM 4
#define STR_LEN 10
#define CHARS 27
#define CONSRV_SIZE 9
#define SEMI_CONSRV_SIZE 11
#define SEQ1_SIZE 3000
#define SEQ2_SIZE 2000
#define MASTER_PROCESS 0
#define THREADS 2

#include <omp.h>

typedef struct Payload {
	char seq1[SEQ1_SIZE + 1];
	char seq2[SEQ2_SIZE + 1];
	int len;
	int max_offset;
} Payload;

typedef struct Alignments {
	int offset;	/* seq2 offset */
	int char_idx;
	int char_val;
	int alignment_score;
} Alignment;

int not_opt(Alignment *a);

void build_table();
void insert_string(const char *str, const char sign);
void output_print(Payload *data, Alignment *res);

Alignment* find_mutants(const Payload *source, Alignment *res, int idx);
Alignment* find_offset(const Payload *source, Alignment *res);
Alignment* compare_and_swap(const Alignment *a1, Alignment *a2);
Alignment* copy(const Alignment *source, Alignment *dest);
Alignment* compare(const Payload *d, Alignment *a);

void get_data(Payload *data, Alignment *res, int * num_of_sequences);

Alignment* computeOnGPU(Payload *source, Alignment *result, char *chars_comparision,
		double *weights, int from, int to);
#endif /* SEQUENCE_ALIGNMENT_H_ */
