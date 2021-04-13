#include <bits/stdc++.h>
#include <ctime>
#include <float.h>
#include "lsm_cpu.h"

using namespace std;

int main(int argc, char **argv){
	if(argc < 2) {
    	cout<<"Usage: " << argv[0] << "<insert/search/delete/count> (choose one)\n";
	    return 1;
    }
    struct timespec start_cpu, end_cpu;
	float msecs_cpu;
	clock_gettime(CLOCK_MONOTONIC, &start_cpu);
	string input = argv[1];
	std::vector<float> v;

	if(input == "insert"){
		lsm_tree* tree = make_lsm_tree();
		for(int i=1;i<=500;i++){
			struct timespec start_cpu_op, end_cpu_op;
			float msecs_cpu_op;
			clock_gettime(CLOCK_MONOTONIC, &start_cpu_op);

			// Operation goes here //
			insert_lsm(i, 1, tree);
			// // // // // // // // //

			clock_gettime(CLOCK_MONOTONIC, &end_cpu_op);
			msecs_cpu_op = 1000.0 * (end_cpu_op.tv_sec - start_cpu_op.tv_sec) + (end_cpu_op.tv_nsec - start_cpu_op.tv_nsec)/1000000.0;
			if(i%10 == 0)
				v.push_back(msecs_cpu_op);
		}
	}
	else if(input == "search"){

	}
	else if(input == "delete"){

	}
	else if(input == "count"){

	}
	else{
		printf("%s\n", "Invalid Operation, exiting...");
		return 1;
	}

	// for(auto x: v){
	// 	cout<<x<<" ";
	// }
	// cout<<"\n";
	
	clock_gettime(CLOCK_MONOTONIC, &end_cpu);
	msecs_cpu = 1000.0 * (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec)/1000000.0;
	cout<<"\tOverall Computation took "<<msecs_cpu<<" milliseconds.\n"<<flush;
}