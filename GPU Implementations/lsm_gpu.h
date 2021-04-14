#ifndef _LSM_GPU_CPP_H
#define _LSM_GPU_CPP_H

#define NUM_OF_TREES 5
#define MEM_ARRAY_SIZE 128  // size of array in memory
#define LEVEL 10 // depth of the log structured tree. 1 level is in main memory, (k-1) levels are on the disk

struct node{
	int key;
	int value;
};

struct lsm_tree{
	int *C_0;
	int ptr_for_c0;
	int *C_i[LEVEL-1];
	int ptr_for_ci[LEVEL-1];
	int max_size_for_ci[LEVEL-1];
};

lsm_tree **get_tree();
void insert_lsm(int key, int val, lsm_tree  **tree);
bool delete_lsm(int key, lsm_tree **tree);

#endif
