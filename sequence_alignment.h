/*

*/

#ifndef SEQUENCE_ALIGNMENT_H_
#define SEQUENCE_ALIGNMENT_H_

#define WEIGHTS_NUM 4
#define CHARS 26
#define FIRST_TYPE_GROUP_SIZE 9
#define SECOND_TYPE_GROUP_SIZE 11
#define SEQ1_SIZE 3000
#define SEQ2_SIZE 2000
#define GROUPS_STRINGS_SIZE 10
#define MASTER_PROCESS 0
#define A_ASCII 'A'
#define RESULT_TAG	71
#include <omp.h>

typedef struct Payload {
	char seq1[SEQ1_SIZE + 1];
	char seq2[SEQ2_SIZE + 1];
	int len;
	int max_offset;
} Payload;

typedef struct Score {
	int offset;	/* seq2 offset */
	int hyphen_idx;	/* hyphen offset */
	int alignment_score;
	int max_score;	/* max score - the length of sequence 2 * first weight */
} Score;

int is_score_optimized(Score *score);

void build_table();
void insert_string(const char *str, const char sign);
void results_output(Score *score, int num_sequences);
void compare_scores_and_swap(const Score *s1, Score *s2);
void calculate_score(const Payload *d, Score *score);
void get_data(Payload *data, Score *score, int * num_of_sequences);

Score* find_optimal_offset_omp(const Payload *source, Score *score, int start_offset, int end_offset);
Score* find_optimal_offset(const Payload *source, Score *score, int start_offset, int end_offset);
Score* deep_copy_score(const Score *source, Score *dest);
Score* cuda_calculation(Payload *source, Score *score, char *chars_comparision,
		int *weights, int start_offset, int end_offset);

#endif /* SEQUENCE_ALIGNMENT_H_ */
