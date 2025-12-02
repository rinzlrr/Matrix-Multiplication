// Joshua Cano & Moses Madale
// User 09 & 05
// GCSC 562
// Lab 2 HPC
// 4/2/25

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <malloc.h>

//******************************************************************************

int n;              // Global matrix dimension (square matrices)
int do_transpose;

//******************************************************************************

double
msDiffTime(struct timespec start, struct timespec finish) {
    long seconds = finish.tv_sec - start.tv_sec; 
    long ns = finish.tv_nsec - start.tv_nsec; 
	    
    if (start.tv_nsec > finish.tv_nsec) { // clock underflow 
	--seconds; 
	ns += 1000000000; 
    } 
    return(1000.0*((double)seconds + (double)ns/(double)(1000000000)));
}

//******************************************************************************

double **allocArray(int r, int c) {
    double **rc, *b;
    size_t sizeofArray = (size_t)r * (size_t)c * (size_t)sizeof(double);

    //fprintf(stderr, "sizeofArray is %ld\n", sizeofArray);
    b = (double *)malloc(sizeofArray);

    rc = (double **)malloc(r * sizeof(double *));

    rc[0] = b;
    for (int i = 1; i < r; i++) {
	rc[i] = rc[i-1] + c;
    }

    return rc;
}

//******************************************************************************

void transpose(double **x) {
    int r, c;
    double t;

    for (r = 0; r < n; r++) {
	for (c = 0; c < n; c++) {
	    t = x[c][r];
	    x[c][r] = x[r][c];
	    x[r][c] = t;
	}
    }
}

//******************************************************************************

double dotProduct(double *a, double *b) {
    int i;
    double dp = 0.0;
    
    for (i = 0; i < n; i++) {
        dp += a[i] * b[i];
    }
    return dp;
}

//******************************************************************************

void mm(double **A, double **B, double **C, int rowCount) {
    int i, j, k;
    if (do_transpose) {
        transpose(B);
        #pragma omp parallel for private(j,k)
        for (i = 0; i < rowCount; i++) {
            for (j = 0; j < n; j++) {
                double sum = 0.0;
                for (k = 0; k < n; k++) {
                    sum += A[i][k] * B[j][k];
                }
                C[i][j] = sum;
            }
        }
        transpose(B);
    } else {
        #pragma omp parallel for private(j,k)
        for (i = 0; i < rowCount; i++) {
            for (j = 0; j < n; j++) {
                double sum = 0.0;
                for (k = 0; k < n; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
    }
}


//********************************************************************************

int main(int argc, char** argv) {
    int nid;
    int nCount;
    int myRowCount, smallRowCount, largeRowCount, rem;
    size_t sizeofArrays;

    double **A, **B, **C;
    int i, j;
    struct timespec tStart, tStop;
    int match_result, global_result;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &nid);
    MPI_Comm_size(MPI_COMM_WORLD, &nCount);

    if (argc < 3) {
	    if (nid == 0) {
	        fprintf(stderr, "Usage: %s n t -- where n is the size of the matrix\n", argv[0]);
	        fprintf(stderr, "   and t is 0 for no-transpose and 1 for transpose\n");
	    }
	    MPI_Finalize();
	    exit(-1);
    }

    sscanf(argv[1], "%d", &n);
    sscanf(argv[2], "%d", &do_transpose);

    if (nid == 0) {
        fprintf(stdout, "n = %d\n", n);
        fprintf(stdout, "nCount = %d\n", nCount);
    }

    // calculate row counts
    rem = n % nCount;
    smallRowCount = n / nCount;
    largeRowCount = smallRowCount + 1;
    
    if (nid < rem) {
        myRowCount = largeRowCount;
    } else {
        myRowCount = smallRowCount;
    }

    // allocate memory for arrays A, B, C
    B = allocArray(n, n);

    if (nid == 0) {
        A = allocArray(n, n);
        C = allocArray(n, n);
        
        // Initialize A with 4.0
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                A[i][j] = 4.0;
            }
        }
        
        // Initialize B as an identity matrix
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                B[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
        
        // Initialize C with -1.0
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                C[i][j] = -1.0;
            }
        }
    } else {
        A = allocArray(myRowCount, n);
        C = allocArray(myRowCount, n);
    }

    if (nid == 0) {
        clock_gettime(CLOCK_MONOTONIC, &tStart);
    }

    // Broadcast B to all nodes
    for (i = 0; i < n; i++) {
        MPI_Bcast(B[i], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (nid == 0) {
        // Node 0 computes its portion
        // Then send portions of A to other nodes
        int offset = myRowCount;
        for (int dest = 1; dest < nCount; dest++) {
            int destRows = (dest < rem) ? largeRowCount : smallRowCount;
            for (i = 0; i < destRows; i++) {
                MPI_Send(A[offset + i], n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
            offset += destRows;
        }
    } else {
        // Receive my portion of A from node 0
        for (i = 0; i < myRowCount; i++) {
            MPI_Recv(A[i], n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // do the matrix multiplication
    mm(A, B, C, myRowCount);

    // determine if my part of the multiplication was correct
    match_result = 1;  // 1 for good result
    for (i = 0; i < myRowCount; i++) {
        for (j = 0; j < n; j++) {
            if (C[i][j] != A[i][j]) {
                match_result = 2;  // 2 for bad result
                break;
            }
        }
        if (match_result == 2) break;
    }

    // Use MPI_Reduce to check if any node had incorrect results
    MPI_Reduce(&match_result, &global_result, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (nid == 0) {
        clock_gettime(CLOCK_MONOTONIC, &tStop);
        
        char hostname[HOST_NAME_MAX];
        gethostname(hostname, HOST_NAME_MAX);
        
        fprintf(stdout, "Base %s with %d nodes for matrix size %d, Time to do mm: %6.5f minutes\n", 
                hostname, nCount, n, msDiffTime(tStart, tStop)/60000.0);

        if (global_result == 1) {
            fprintf(stdout, "Matrix calculation is correct\n");
        } else {
            fprintf(stdout, "Matrix calculation is INCORRECT\n");
        }
    }

    MPI_Finalize();
    return 0;

}