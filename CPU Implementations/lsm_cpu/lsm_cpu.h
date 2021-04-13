#ifndef _SDT_GPU_H_
#define _SDT_GPU_H_

#define MEM_ARRAY_SIZE 128  // size of array in memory
#define LEVEL 7 // depth of the log structured tree. 1 level is in main memory, (k-1) levels are on the disk

struct node{
	int key;
	int value;
};

struct lsm_tree{
	int C_0[MEM_ARRAY_SIZE];
	int ptr_for_c0;
	int *C_i[LEVEL-1];
	int ptr_for_ci[LEVEL-1];
	int max_size_for_ci[LEVEL-1];
};

lsm_tree* make_lsm_tree();
int* merge(int *a, int *b, int n1, int n2);
void insert_lsm(int key, int value, lsm_tree* tree);
bool search_in_lsm(int key, lsm_tree* tree);
int count(int k1, int k2, lsm_tree* tree);
bool delete_key(int key, lsm_tree* tree);

#endif
