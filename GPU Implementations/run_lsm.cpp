#include <cuda_runtime.h>
#include <iostream>
#include "lsm_gpu.h"
#include <bits/stdc++.h>
using namespace std;

#define ELEMENTS 100000

extern "C" void insert_lsm_helper(int key, int val, lsm_tree **tree);
extern "C" bool delete_lsm_helper(int key, lsm_tree **tree);
extern "C" lsm_tree **get_tree_from_cuda();

lsm_tree **get_tree(){
	return get_tree_from_cuda();
}

void insert_lsm(int key, int val, lsm_tree **tree)
{
  insert_lsm_helper(key, val, tree);
}

bool delete_lsm(int key, lsm_tree **tree)
{
  return delete_lsm_helper(key, tree);
}

int main(){

  lsm_tree **tree = get_tree();
  cudaEvent_t start_gpu, end_gpu;
  float msecs_gpu;

  cudaEvent_t start_gpu_one, end_gpu_one;
  float msecs_gpu_one;

  cudaEventCreate(&start_gpu);
  cudaEventCreate(&end_gpu);
  cudaEventRecord(start_gpu, 0);

  vector<float> v;

  for(int i=0;i<ELEMENTS;i++){
    cudaEventCreate(&start_gpu_one);
    cudaEventCreate(&end_gpu_one);
    cudaEventRecord(start_gpu_one, 0);
    insert_lsm(i, 1, tree);
    cudaEventRecord(end_gpu_one, 0);
    cudaEventSynchronize(end_gpu_one);
    cudaEventElapsedTime(&msecs_gpu_one, start_gpu_one, end_gpu_one);
    cudaEventDestroy(start_gpu_one);
    cudaEventDestroy(end_gpu_one);
    if(i%50 == 0){
      v.push_back(msecs_gpu_one);
    }
  }
  cudaEventRecord(end_gpu, 0);
  cudaEventSynchronize(end_gpu);
  cudaEventElapsedTime(&msecs_gpu, start_gpu, end_gpu);
  cudaEventDestroy(start_gpu);
  cudaEventDestroy(end_gpu);
  cout<<"\tInsertion took "<<msecs_gpu/ELEMENTS<<" milliseconds.\n";

  cudaEventCreate(&start_gpu);
  cudaEventCreate(&end_gpu);
  cudaEventRecord(start_gpu, 0);

  ofstream outputFile ("insert_vec.txt");
  outputFile << "";
  copy(v.begin(), v.end(), ostream_iterator<float>(outputFile , ", ")); 
  outputFile << "" << endl;

  vector<float> v1;

  for(int i=0;i<ELEMENTS;i++){
    cudaEventCreate(&start_gpu_one);
    cudaEventCreate(&end_gpu_one);
    cudaEventRecord(start_gpu_one, 0);
    delete_lsm(i, tree);
    cudaEventRecord(end_gpu_one, 0);
    cudaEventSynchronize(end_gpu_one);
    cudaEventElapsedTime(&msecs_gpu_one, start_gpu_one, end_gpu_one);
    cudaEventDestroy(start_gpu_one);
    cudaEventDestroy(end_gpu_one);
    if(i%50 == 0){
      v1.push_back(msecs_gpu_one);
    }
  }
  cudaEventRecord(end_gpu, 0);
  cudaEventSynchronize(end_gpu);
  cudaEventElapsedTime(&msecs_gpu, start_gpu, end_gpu);
  cudaEventDestroy(start_gpu);
  cudaEventDestroy(end_gpu);
  cout<<"\tDeletion took "<<msecs_gpu/ELEMENTS<<" milliseconds.\n";

  ofstream outputFile1 ("delete_vec.txt");
  outputFile1 << "";
  copy(v1.begin(), v1.end(), ostream_iterator<float>(outputFile1 , ", ")); 
  outputFile1 << "" << endl;
}