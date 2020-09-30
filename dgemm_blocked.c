const char* dgemm_desc = "Simple blocked dgemm with copy optimization.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 16)
#endif

/*
  A is M-by-K
  B is K-by-N
  C is M-by-N

  lda is the leading dimension of the matrix (the M of square_dgemm).
*/
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double* restrict A, const double* restrict B, double *C)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            double cij = C[j*lda+i];
            for (k = 0; k < K; ++k) {
                cij += A[k*lda+i] * B[j*lda+k];
            }
            C[j*lda+i] = cij;
        }
    }
}

void do_block(const int lda,
              const double* restrict A, const double* restrict B, double *C,
              const int i, const int j)
{
    const int M = BLOCK_SIZE;
    const int N = BLOCK_SIZE;
    const int K = BLOCK_SIZE;
    basic_dgemm(lda, M, N, K,
                A, B, C + i + j*lda);
}
/*
void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bi = 0; bi < n_blocks; ++bi) {
        const int i = bi * BLOCK_SIZE;
        for (bj = 0; bj < n_blocks; ++bj) {
            const int j = bj * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                const int k = bk * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}
*/
void square_dgemm(const int M, const double *A, const double *B, double *C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    double *blocks = copy_blocks(M, n_blocks, A, B);
    int bi, bj, bk;
    for (bj = 0; bj < n_blocks; ++bj) {
        const int j = bj * BLOCK_SIZE;
        for (bi = 0; bi < n_blocks; ++bi) {
            const int i = bi * BLOCK_SIZE;
            for (bk = 0; bk < n_blocks; ++bk) {
                do_block(M, blocks + 2 * (bk * BLOCK_SIZE * BLOCK_SIZE + bi), blocks + 2 * (bj * BLOCK_SIZE * BLOCK_SIZE + bk) + BLOCK_SIZE * BLOCK_SIZE, C, i, j);
            }            
        }
    }
}

double* copy_blocks(const int M, const int n_blocks, const double* A, const double* B)
{
    double* blocks = (double*)aligned_alloc(8, 1000*BLOCK_SIZE*BLOCK_SIZE*sizeof(double));
    int i, j, bi, bj, ri, rj;
    for (j = 0; j < BLOCK_SIZE * n_blocks; ++j) {
        for (i = 0; i < BLOCK_SIZE * n_blocks; ++i) {
            bi = i / BLOCK_SIZE;
            bj = j / BLOCK_SIZE;
            ri = i % BLOCK_SIZE;
            rj = j % BLOCK_SIZE;
            if (i >= M || j >= M) {
                blocks[2 * (bj * n_blocks + bi) * BLOCK_SIZE * BLOCK_SIZE + rj * BLOCKSIZE + ri] = (double)0;
                blocks[2 * (bj * n_blocks + bi) * BLOCK_SIZE * BLOCK_SIZE + rj * BLOCKSIZE + ri + BLOCK_SIZE * BLOCK_SIZE] = (double)0;
            }
            blocks[2 * (bj * n_blocks + bi) * BLOCK_SIZE * BLOCK_SIZE + rj * BLOCKSIZE + ri] = A[j * M + i];
            blocks[2 * (bj * n_blocks + bi) * BLOCK_SIZE * BLOCK_SIZE + rj * BLOCKSIZE + ri + BLOCK_SIZE * BLOCK_SIZE] = B[j * M + i];
        }
    }
    return blocks;
            
