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
#include <time.h>


#include "sequence_alignment.h"

/* Strings group declaration */
char first_type_groups[FIRST_TYPE_GROUP_SIZE][GROUPS_STRINGS_SIZE] = {"NDEQ", "MILV", "FYW", "NEQK", "QHRK", "HY", "STA", "NHQK", "MILF"};

char second_type_groups[SECOND_TYPE_GROUP_SIZE][GROUPS_STRINGS_SIZE] = {"SAG", "SGND", "NEQHRK", "HFY", "ATV", "STPA", "NDEQHK", "FVLIM", "CSA", "STNK", "SNDEQK"};


int weights[WEIGHTS_NUM];
/* table for storing characters comparision answers */
char chars_comparision[CHARS][CHARS];

int main(int argc, char *argv[]) {
	fprintf(stdout, "Alignment Sequence Algorithm is running...\n");

	int num_of_sequences;
	
	clock_t start, end;
	double cpu_time_used;

	Payload *data;
	Score *scores;

	start = clock();

	/* Handle the input and data initalization */
	build_table();

	/* Get input from stdin */
	char seq1[SEQ1_SIZE];
	fscanf(stdin, "%d %d %d %d", &weights[0], &weights[1], &weights[2],
		&weights[3]);
	/* Get sequence 1 and convert to uppercase */
	fscanf(stdin, "%s", seq1);
	for (int i = 0; i < strlen(seq1); i++) {
		if (seq1[i] >= 'a' && seq1[i] <= 'z')
			seq1[i] = toupper(seq1[i]);
	}
	fscanf(stdin, "%d", &num_of_sequences);
	data = (Payload *)malloc(sizeof(Payload) * num_of_sequences);
	scores = (Score *)malloc(sizeof(Score) * num_of_sequences);
	for (int i = 0; i < num_of_sequences; i++)
	{
		// data[i].seq1 = strdup(seq1);
		strcpy(data[i].seq1, seq1);
		fscanf(stdin, "%s", data[i].seq2);
		data[i].len = strlen(data[i].seq2);
		for (int j = 0; j < data[j].len; j++) {
			if (data[i].seq2[j] >= 'a' && data[i].seq2[j] <= 'z')
				data[i].seq2[j] = toupper(data[i].seq2[j]);
		}
		data[i].max_offset = strlen(seq1) - data[i].len;
		scores[i].hyphen_idx = 0;
		scores[i].offset = 0;
		scores[i].alignment_score = 0;
		scores[i].max_score = strlen(data[i].seq2) * weights[0]; // stop if reached

	}
	
	/* Calcuate optimal offset (n) and mutant (k) */
	for (int i = 0; i < num_of_sequences; i++) {
		find_optimal_offset(&data[i], &scores[i]);
	}
	
	/* Time taken for program */
	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	/* Print results */
	fprintf(stdout, "Execution time is: %lf\n", cpu_time_used);
	results_output(scores, num_of_sequences);

	free(data);
	free(scores);
	
	return 0;
}

Score* find_optimal_mutant(const Payload *source, Score *scores, int idx) {
	Score *tmp = (Score*) malloc(sizeof(Score));
	tmp->hyphen_idx = idx;
	compare(source, tmp);
	compare_scores_and_swap(tmp, scores);

	free(tmp);
	return scores;
}

Score* find_optimal_offset(const Payload *source, Score *score) {

	Score *tmp = (Score*) malloc(sizeof(Score));
	tmp = deep_copy_score(score, tmp);

	compare(source, score); /* hyphen_idx = 0 -> means the hyphen is at the end of ths string (e.g. ABC-) */

	for (int j = 1; j < source->len && is_score_optimized(score); ++j) {	
		score->hyphen_idx = j;
		compare(source, tmp);
		compare_scores_and_swap(tmp, score);
	}	
	for (int i = 1; i <= source->max_offset - 1 && is_score_optimized(score); ++i) {
		tmp->offset = i;
		/* for each offset find optimal mutant */
		for (int j = 1; j < source->len && is_score_optimized(score); ++j) {	
			// tmp = find_optimal_mutant(source, tmp, j);
			tmp->hyphen_idx = j;
			compare(source, tmp);
			compare_scores_and_swap(tmp, score);
		}	
		compare_scores_and_swap(tmp, score);
	}
	if (score->hyphen_idx == 0) {
		score->hyphen_idx = source->len;
	}
	else {
		score->hyphen_idx += 1;
	}
	free(tmp);
	return score;

}

/* The function compares the alignment score and swaps the results if needed */
void compare_scores_and_swap(const Score *s1, Score *s2) {
	if (s1->alignment_score > s2->alignment_score) {
		deep_copy_score(s1, s2);
	}
	// return a2;
}

int is_score_optimized(Score *score) {
	return !(score->alignment_score == score->max_score);
}

Score* deep_copy_score(const Score *source, Score *dest) {
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
void compare(const Payload *payload, Score *score) {
	int seq1_char, seq2_char;
	score->alignment_score = 0;
	for (int char_index = 0; char_index < payload->len; ++char_index) {
		/* Convert char to ABC order index */
		if (char_index == score->hyphen_idx && score->hyphen_idx != 0) {
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
	for (int i = 0; i < FIRST_TYPE_GROUP_SIZE; ++i) {
		insert_string(first_type_groups[i], '%');
	}

	/* Fill second type groups characters */
	for (int i = 0; i < SECOND_TYPE_GROUP_SIZE; ++i) {
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

void results_output(Score *scores, int num_sequences) {
	fprintf(stdout, "The number of sequences in the input is: %d\n", num_sequences);
	for (int i = 0; i < num_sequences; ++i) {
		fprintf(stdout, "Sequence #%d | optimal offset(n): %d | optimal hyphen index(k): %d | alignment score: %d\n", i + 1, scores[i].offset, scores[i].hyphen_idx, scores[i].alignment_score);
	}	
}
