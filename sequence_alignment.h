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

typedef struct Payload {
	char seq1[SEQ1_SIZE + 1];
	char seq2[SEQ2_SIZE + 1];
	int len;
	int max_offset;
} Payload;

typedef struct Alignments {
	int offset;	/* seq2 offset */
	int hyphen_idx;	/* hyphen offset */
	int alignment_score;
	int max_score;
} Alignment;

int not_opt(Alignment *a);

void build_table();
void insert_string(const char *str, const char sign);
void results_output(Alignment *res, int num_sequences);

Alignment* find_optimal_offset(const Payload *source, Alignment *res);
Alignment* copy(const Alignment *source, Alignment *dest);
void compare_and_swap(const Alignment *a1, Alignment *a2);
void compare(const Payload *d, Alignment *a);

#endif /* SEQUENCE_ALIGNMENT_H_ */
