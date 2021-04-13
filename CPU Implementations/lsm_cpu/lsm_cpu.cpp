#include <bits/stdc++.h>
#include "lsm_cpu.h"
using namespace std;

lsm_tree* make_lsm_tree(){
	lsm_tree *tree = (lsm_tree *) malloc(sizeof(lsm_tree));
	memset(tree->C_0, 0, MEM_ARRAY_SIZE*sizeof(int));
	tree->ptr_for_c0 = 0;
	for(int i=0;i<LEVEL-1;i++){
		tree->ptr_for_ci[i] = 0;
		tree->max_size_for_ci[i] = MEM_ARRAY_SIZE*pow(2, i+1);
	}
	return tree;
}

int* merge(int *a, int *b, int n1, int n2){
	int *c = (int *) malloc(sizeof(int)*(n1+n2));
	int i = 0, j = 0, k = 0;
	while(i<n1 || j<n2){
		if(i>n1-1){
			c[k++] = b[j++];
		}
		else if(j>n2-1){
			c[k++] = a[i++];
		}
		else if(a[i]<b[j]){
			c[k++] = a[i++];
		}
		else{
			c[k++] = b[j++];
		}
	}
	return c;
}

void insert_lsm(int key, int value, lsm_tree* tree){
	int node = key;
	int i = tree->ptr_for_c0;
	while(i>=0 && tree->C_0[i] > key){
		tree->C_0[i+1] = tree->C_0[i];
	}
	tree->C_0[i] = key;
	tree->ptr_for_c0 += 1;

	if(tree->ptr_for_c0 == MEM_ARRAY_SIZE){
		// Write to the array on disk
		int i=0;
		while(i<LEVEL-1 && tree->ptr_for_ci[i] >= tree->max_size_for_ci[i]){
			int *new_tree =  merge(tree->C_i[i], tree->C_i[i+1], tree->ptr_for_ci[i], tree->ptr_for_ci[i+1]);
			free(tree->C_i[i+1]);
			tree->C_i[i+1] = new_tree;
			tree->ptr_for_ci[i+1] += tree->ptr_for_ci[i];
			tree->ptr_for_ci[i] = 0;
			memset(tree->C_i[i], 0, tree->max_size_for_ci[i]*sizeof(int));
			i+=1;	
		}
		int *new_tree =  merge(tree->C_i[0], tree->C_0, tree->ptr_for_ci[0], MEM_ARRAY_SIZE);
		free(tree->C_i[0]);
		tree->C_i[0] = new_tree;
		tree->ptr_for_ci[0] += MEM_ARRAY_SIZE;
		tree->ptr_for_c0 = 0;
		memset(tree->C_0, 0, MEM_ARRAY_SIZE*sizeof(int));
	}
}

bool search_in_lsm(int key, lsm_tree* tree){
	if((binary_search(tree->C_0, tree->C_0 + tree->ptr_for_c0, key)))
		return true;
	for(int i=0;i<LEVEL-1;i++){
		if(tree->ptr_for_ci[i]>0)
			if(binary_search(tree->C_i[i], tree->C_i[i] + tree->ptr_for_ci[i], key))
				return true;
	}
	return false;
}

int count(int k1, int k2, lsm_tree* tree){
	int no_of_elements = 0;
	int lower_index = -1;
	int upper_index = -1;
	if(tree->ptr_for_c0 > 0){
		lower_index = lower_bound(tree->C_0, tree->C_0 + tree->ptr_for_c0, k1) - tree->C_0;
		upper_index = upper_bound(tree->C_0, tree->C_0 + tree->ptr_for_c0, k2) - tree->C_0;
		no_of_elements += (upper_index - lower_index + 1);

		if(upper_index == tree->ptr_for_c0)
			no_of_elements--;
	}
	for(int i=0; i<LEVEL-1;i++){
		if(tree->ptr_for_ci[i] > 0){
			lower_index = lower_bound(tree->C_i[i], tree->C_i[i] + tree->ptr_for_ci[i], k1) - tree->C_i[i];
			upper_index = upper_bound(tree->C_i[i], tree->C_i[i] + tree->ptr_for_ci[i], k2) - tree->C_i[i];
			no_of_elements += (upper_index - lower_index + 1);

			if(upper_index == tree->ptr_for_ci[i])
				no_of_elements--;
		}
	}
	return no_of_elements;
}

bool delete_key(int key, lsm_tree* tree){
	// deletes the first occurence of the key (latest)
	int upper_index = -1;
	if(tree->ptr_for_c0 > 0){
		upper_index = upper_bound(tree->C_0, tree->C_0 + tree->ptr_for_c0, key) - tree->C_0;
		upper_index--;
		if(tree->C_0[upper_index] == key){
			for(int i=upper_index; i<tree->ptr_for_c0-1;i++){
				tree->C_0[i] = tree->C_0[i+1];
			}
			tree->ptr_for_c0--;
			return true;
		}
	}
	for(int i=0;i<LEVEL-1;i++){
		if(tree->ptr_for_ci[i] > 0){
			upper_index = upper_bound(tree->C_i[i], tree->C_i[i] + tree->ptr_for_ci[i], key) - tree->C_i[i];
			upper_index--;
			if(tree->C_i[i][upper_index] == key){
				for(int j=upper_index; j<tree->ptr_for_ci[i]-1;j++){
					tree->C_i[i][j] = tree->C_i[i][j+1];
				}
				tree->ptr_for_ci[i]--;
				return true;
			}
		}
	}
	return false;
}

// int main(){
// 	lsm_tree* tree = make_lsm_tree();
// 	for(int i=1;i<=100;i++){
// 		insert_lsm(i, 1, tree);
// 	}
// 	cout<<search_in_lsm(1, tree)<<"\n";
// 	cout<<search_in_lsm(42, tree)<<"\n";
// 	cout<<search_in_lsm(234, tree)<<"\n";
// 	cout<<search_in_lsm(420, tree)<<"\n";
// 	cout<<count(-1, 400, tree)<<"\n";
// 	cout<<delete_key(102, tree)<<"\n";
// 	for(int i=101;i<=383;i++){
// 		insert_lsm(i, 1, tree);
// 	}
// 	cout<<delete_key(383, tree)<<"\n";
// 	for (int i = 0; i < tree->ptr_for_c0; ++i)
// 	{
// 		printf("%d ", tree->C_0[i]);
// 	}
// 	cout<<"\n";
// 	cout<<tree->ptr_for_c0<<"/"<<MEM_ARRAY_SIZE<<"\n";
// 	for(int i=0;i<LEVEL-1;i++)
// 		cout<<tree->ptr_for_ci[i]<<"/"<<tree->max_size_for_ci[i]<<"\n";
// }