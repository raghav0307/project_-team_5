#include <stdio.h>
#include <float.h>
#include "lsm_gpu.h"

#define THREADS_PER_BLOCK 8

__global__ void bitonicSortKernel(int *arr, int j, int k, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int l = i ^ j;
  if (i < l && l < n) {
    int dir = i & k;
    if ((dir == 0 && arr[i] > arr[l]) || (dir != 0 && arr[i] < arr[l])) {
      int temp = arr[i];
      arr[i] = arr[l];
      arr[l] = temp;
    }
  }
}

__host__ __device__ void bitonicSortPower2(int *array, int n) {
  int threads = n / 2;
  int blocks = (threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  for (int k = 2; k <= n; k *= 2)
    for (int j = k / 2; j > 0; j /= 2)
      bitonicSortKernel<<<blocks, THREADS_PER_BLOCK>>>(array, j, k, n);
	  cudaDeviceSynchronize();
}

__host__ __device__ int maxPowerLessThan(int n) {
  int ret = 1;
  n = n / 2;
  while (n) {
    ret *= 2;
    n /= 2;
  }
  return ret;
}

__host__ __device__ void bitonicSort(int *array, int n) {
  int len = maxPowerLessThan(n);
  bitonicSortPower2(array, len);
  bitonicSortPower2(&array[n - len], len);
  int offset = (n - len) / 2;
  bitonicSortPower2(&array[offset], len);
  bitonicSortPower2(array, len);
  bitonicSortPower2(&array[n - len], len);
  bitonicSortPower2(&array[offset], len);
}
__global__ void append_to_array(int* a, int *b, int off, int ele){
	int i = (threadIdx.x + blockDim.x*blockIdx.x);
	if(i < ele)
		a[i + off] = b[i];
}

__global__ void insert_in_lsm(int key, int value, lsm_tree **tree, int bucket){
	int i = tree[bucket]->ptr_for_c0;
	while(i>0 && tree[bucket]->C_0[i] > key){
		tree[bucket]->C_0[i+1] = tree[bucket]->C_0[i];
		i--;
	}
	tree[bucket]->C_0[i] = key;
	tree[bucket]->ptr_for_c0 += 1;
	if(tree[bucket]->ptr_for_c0 == MEM_ARRAY_SIZE){
		int i=0;
		while(i<LEVEL-1 && tree[bucket]->ptr_for_ci[i] >= tree[bucket]->max_size_for_ci[i]){
			append_to_array<<<tree[bucket]->ptr_for_ci[i]/32, 32>>>(tree[bucket]->C_i[i+1], tree[bucket]->C_i[i], tree[bucket]->ptr_for_ci[i+1], tree[bucket]->ptr_for_ci[i]);
			cudaDeviceSynchronize();
			tree[bucket]->ptr_for_ci[i+1] += tree[bucket]->ptr_for_ci[i];
			tree[bucket]->ptr_for_ci[i] = 0;
			bitonicSort(tree[bucket]->C_i[i+1], tree[bucket]->ptr_for_ci[i+1]);
			i+=1;	
		}
		append_to_array<<<tree[bucket]->ptr_for_ci[0]/32, 32>>>(tree[bucket]->C_i[0], tree[bucket]->C_0, tree[bucket]->ptr_for_ci[0], tree[bucket]->ptr_for_c0);
		cudaDeviceSynchronize();
		tree[bucket]->ptr_for_ci[0] += tree[bucket]->ptr_for_c0;
		tree[bucket]->ptr_for_c0 = 0;
		bitonicSort(tree[bucket]->C_i[0], tree[bucket]->ptr_for_ci[0]);
	}
}

__global__ void search(int *a, int n, int key, int *index){
	int i = (threadIdx.x + blockDim.x*blockIdx.x);
	if(i<n){
		if(a[i]==key)
			*index = i;
	}
}

__global__ void delete_key(int key, lsm_tree **tree, bool *result, int *index){
	int bucket = blockIdx.x;
	int level = threadIdx.x;
	if(bucket < NUM_OF_TREES && level < LEVEL){
		if(level == 0){
			if(tree[bucket]->ptr_for_c0 > 0){
				search<<<tree[bucket]->ptr_for_c0/32, 32>>>(tree[bucket]->C_0, tree[bucket]->ptr_for_c0, key, index);
				cudaDeviceSynchronize();
				if(*index != -1){
					for(int i=*index; i<tree[bucket]->ptr_for_c0-1;i++){
						tree[bucket]->C_0[i] = tree[bucket]->C_0[i+1];
					}
					tree[bucket]->ptr_for_c0--;
					*result = true;
				}
			}
		}
		else{
			level--;
			if(tree[bucket]->ptr_for_ci[level] > 0){
				search<<<tree[bucket]->ptr_for_ci[level]/32, 32>>>(tree[bucket]->C_i[level], tree[bucket]->ptr_for_ci[level], key, index);
				cudaDeviceSynchronize();
				if(*index != -1){
					for(int j=*index; j<tree[bucket]->ptr_for_ci[level]-1;j++){
						tree[bucket]->C_i[level][j] = tree[bucket]->C_i[level][j+1];
					}
					tree[bucket]->ptr_for_ci[level]--;
					*result = true;
				}
			}
		}
	}
}

__global__ void search_in_lsm(int key, lsm_tree** tree, bool* result){
	;
}

__global__ void count_in_lsm(int k1, int k2, lsm_tree** tree, int count){
	;
}

__global__ void init(lsm_tree** tree){
	int i = threadIdx.x;
	int bucket = blockIdx.x;
	if(bucket< NUM_OF_TREES && i<LEVEL){
		cudaMalloc((void**)&tree[bucket], NUM_OF_TREES*sizeof(lsm_tree));
		cudaMalloc((void**)&tree[bucket]->C_0, MEM_ARRAY_SIZE*sizeof(int));
		memset(tree[bucket]->C_0, 0, MEM_ARRAY_SIZE*sizeof(int));
		tree[bucket]->ptr_for_c0 = 0;
		if(i<LEVEL-1){
			tree[bucket]->ptr_for_ci[i] = 0;
			tree[bucket]->max_size_for_ci[i] = MEM_ARRAY_SIZE*pow(2, i+1);
			cudaMalloc((void**)&tree[bucket]->C_i[i], MEM_ARRAY_SIZE*pow(2, i+1)*sizeof(int));
			memset(tree[bucket]->C_i[i], 0, tree[bucket]->max_size_for_ci[i]*sizeof(int));
		}
	}
}

extern "C" lsm_tree **get_tree_from_cuda(){
	lsm_tree **tree;
	cudaMalloc((void**)&tree, NUM_OF_TREES*sizeof(lsm_tree *));
	init<<<NUM_OF_TREES, LEVEL-1>>>(tree);	
	return tree;
}

extern "C" void insert_lsm_helper(int key, int val, lsm_tree **tree){
	// call kernel here
	int bucket = key%NUM_OF_TREES;
	insert_in_lsm<<<1, 1>>>(key, val, tree, bucket);
	cudaDeviceSynchronize();
}

extern "C" bool delete_lsm_helper(int key, lsm_tree **tree){
	// call kernel here
	int *index_d;
	bool *ret_d;
	cudaMalloc((void**)&index_d, sizeof(int));
	cudaMalloc((void**)&ret_d, sizeof(bool));
	cudaDeviceSynchronize();
	delete_key<<<NUM_OF_TREES, LEVEL>>>(key, tree, ret_d, index_d);
	bool *ret_h = (bool *) malloc(sizeof(bool));
	cudaMemcpy(ret_d, ret_h, sizeof(bool), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	return *ret_h;
}