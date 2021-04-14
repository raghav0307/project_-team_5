#include <iostream>

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

void bitonicSortPower2(int *array, int n) {
  int threads = n / 2;
  int blocks = (threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  for (int k = 2; k <= n; k *= 2)
    for (int j = k / 2; j > 0; j /= 2)
      bitonicSortKernel<<<blocks, THREADS_PER_BLOCK>>>(array, j, k, n);
}

int maxPowerLessThan(int n) {
  int ret = 1;
  n = n / 2;
  while (n) {
    ret *= 2;
    n /= 2;
  }
  return ret;
}

void bitonicSort(int *array, int n) {
  int len = maxPowerLessThan(n);
  bitonicSortPower2(array, len);
  bitonicSortPower2(&array[n - len], len);
  int offset = (n - len) / 2;
  bitonicSortPower2(&array[offset], len);
  bitonicSortPower2(array, len);
  bitonicSortPower2(&array[n - len], len);
  bitonicSortPower2(&array[offset], len);
}

int main() {
  int n = 15;
  int *h_arr;
  h_arr = (int *)malloc(n * sizeof(int));
  for (int i = 0; i < n; i++) {
    h_arr[i] = n - i;
  }
  int *d_arr;
  cudaMalloc((void **)&d_arr, n * sizeof(int));
  cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);
  bitonicSort(d_arr, n);
  cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) {
    std::cout << h_arr[i] << " ";
  }
}