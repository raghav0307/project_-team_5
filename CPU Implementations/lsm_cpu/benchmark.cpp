#include <bits/stdc++.h>
#include <ctime>
#include <float.h>
#include "lsm_cpu.h"

using namespace std;

#define ELEMENTS 100000

int main(int argc, char **argv){
	if(argc < 2) {
    	cout<<"Usage: " << argv[0] << "<insert/search/delete/count> (choose one)\n";
	    return 1;
    }
    
	string input = argv[1];
	std::vector<float> v;
	
	if(input == "insert"){
		lsm_tree* tree = make_lsm_tree();
		struct timespec start_cpu, end_cpu;
		float msecs_cpu;
		clock_gettime(CLOCK_MONOTONIC, &start_cpu);
		
		for(int i=1;i<=ELEMENTS;i++){
			struct timespec start_cpu_op, end_cpu_op;
			float msecs_cpu_op;
			clock_gettime(CLOCK_MONOTONIC, &start_cpu_op);

			// Operation goes here //
			insert_lsm(i, 1, tree);
			// // // // // // // // //

			clock_gettime(CLOCK_MONOTONIC, &end_cpu_op);
			msecs_cpu_op = 1000.0 * (end_cpu_op.tv_sec - start_cpu_op.tv_sec) + (end_cpu_op.tv_nsec - start_cpu_op.tv_nsec)/1000000.0;
			if(i%50 == 0)
				v.push_back(msecs_cpu_op);
		}

		clock_gettime(CLOCK_MONOTONIC, &end_cpu);
		msecs_cpu = 1000.0 * (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec)/1000000.0;
		cout<<"\tOverall Computation took "<<msecs_cpu/ELEMENTS<<" milliseconds.\n"<<flush;

	  ofstream outputFile ("insert_vec.txt");
	  outputFile << "";
	  copy(v.begin(), v.end(), ostream_iterator<float>(outputFile , ", ")); 
	  outputFile << "" << endl;

	}
	else if(input == "search"){
		struct timespec start_cpu, end_cpu;
		float msecs_cpu;
		clock_gettime(CLOCK_MONOTONIC, &start_cpu);

		lsm_tree* tree = make_lsm_tree();
		for(int i=1;i<=ELEMENTS;i++){
			insert_lsm(i, 1, tree);
		}

		for(int i=1;i<=ELEMENTS;i++){
			struct timespec start_cpu_op, end_cpu_op;
			float msecs_cpu_op;
			clock_gettime(CLOCK_MONOTONIC, &start_cpu_op);

			// Operation goes here //
			search_in_lsm(i, tree);
			// // // // // // // // //

			clock_gettime(CLOCK_MONOTONIC, &end_cpu_op);
			msecs_cpu_op = 1000.0 * (end_cpu_op.tv_sec - start_cpu_op.tv_sec) + (end_cpu_op.tv_nsec - start_cpu_op.tv_nsec)/1000000.0;
			if(i%50 == 0)
				v.push_back(msecs_cpu_op);
		}

		clock_gettime(CLOCK_MONOTONIC, &end_cpu);
		msecs_cpu = 1000.0 * (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec)/1000000.0;
		cout<<"\tOverall Computation took "<<msecs_cpu/ELEMENTS<<" milliseconds.\n"<<flush;

	  ofstream outputFile ("search_vec.txt");
	  outputFile << "";
	  copy(v.begin(), v.end(), ostream_iterator<float>(outputFile , ", ")); 
	  outputFile << "" << endl;
	}
	else if(input == "delete"){
		struct timespec start_cpu, end_cpu;
		float msecs_cpu;
		clock_gettime(CLOCK_MONOTONIC, &start_cpu);
		lsm_tree* tree = make_lsm_tree();
		for(int i=1;i<=ELEMENTS;i++){
			insert_lsm(i, 1, tree);
		}

		for(int i=1;i<=ELEMENTS;i++){
			struct timespec start_cpu_op, end_cpu_op;
			float msecs_cpu_op;
			clock_gettime(CLOCK_MONOTONIC, &start_cpu_op);

			// Operation goes here //
			delete_key(i, tree);
			// // // // // // // // //

			clock_gettime(CLOCK_MONOTONIC, &end_cpu_op);
			msecs_cpu_op = 1000.0 * (end_cpu_op.tv_sec - start_cpu_op.tv_sec) + (end_cpu_op.tv_nsec - start_cpu_op.tv_nsec)/1000000.0;
			if(i%50 == 0)
				v.push_back(msecs_cpu_op);
		}

		clock_gettime(CLOCK_MONOTONIC, &end_cpu);
		msecs_cpu = 1000.0 * (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec)/1000000.0;
		cout<<"\tOverall Computation took "<<msecs_cpu/ELEMENTS<<" milliseconds.\n"<<flush;
	  

	  ofstream outputFile ("delete_vec.txt");
	  outputFile << "";
	  copy(v.begin(), v.end(), ostream_iterator<float>(outputFile , ", ")); 
	  outputFile << "" << endl;

	}
	else if(input == "count"){
		struct timespec start_cpu, end_cpu;
		float msecs_cpu;
		clock_gettime(CLOCK_MONOTONIC, &start_cpu);

		lsm_tree* tree = make_lsm_tree();
		for(int i=1;i<=ELEMENTS;i++){
			insert_lsm(i, 1, tree);
		}
	
		for(int i=1;i<=ELEMENTS;i++){
			struct timespec start_cpu_op, end_cpu_op;
			float msecs_cpu_op;
			clock_gettime(CLOCK_MONOTONIC, &start_cpu_op);

			// Operation goes here //
			count(i, i+2412,tree);
			// // // // // // // // //

			clock_gettime(CLOCK_MONOTONIC, &end_cpu_op);
			msecs_cpu_op = 1000.0 * (end_cpu_op.tv_sec - start_cpu_op.tv_sec) + (end_cpu_op.tv_nsec - start_cpu_op.tv_nsec)/1000000.0;
			if(i%50 == 0)
				v.push_back(msecs_cpu_op);
		}

		clock_gettime(CLOCK_MONOTONIC, &end_cpu);
		msecs_cpu = 1000.0 * (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_nsec - start_cpu.tv_nsec)/1000000.0;
		cout<<"\tOverall Computation took "<<msecs_cpu/ELEMENTS<<" milliseconds.\n"<<flush;

	  ofstream outputFile ("count_vec.txt");
	  outputFile << "";
	  copy(v.begin(), v.end(), ostream_iterator<float>(outputFile , ", ")); 
	  outputFile << "" << endl;
	}
	else{
		printf("%s\n", "Invalid Operation, exiting...");
		return 1;
	}
}