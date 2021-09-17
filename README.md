# Search Optimised Index based on Log structured Merge Trees

## Team No: 5

* **Raghav Gupta** - 2018076
* **Suchet Aggarwal** - 2018105

## Introduction

Most modern databases deal with enormous amounts of data, with high frequency of data insertions and look ups. Handling such enormous data with traditional methods is not time efficient and thus databases are organised in indexes for fast data retrieval. Most modern day database systems implement these in the form of B-Trees, Fractal trees or Log structured merge trees. The trade off one needs to make is then dependant on the particular use case, whether one wants efficient performance for search or for insertions depending upon the frequency of retrievals and updates. 

In this project we aim at comparing the serial implementations of these indexing methods with one another and then identify the the scope of parallelizing log structured merge trees to compare the speedups one can achieve over the best serial indexing method. 


## Results

| | Insertion | Deletion |
|---|---|---|---|---|
| CPU (B-Trees) | 0.000337428 | 0.00167183 |
| CPU (LSM Trees) | 0.00049103 | 0.0058738 |
| GPU | 0.00030427 | 0.00123993 |

### Speedup

| | Insertion | Deletion |
|---|---|---|---|---|
| GPU vs CPU (B-Trees) | 1.11x | 1.30x |
| GPU vs CPU (LSM Trees) | 1.61x | 4.73x |