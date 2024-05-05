#include "stencil/solve.h"

#include <assert.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>

// NOTE: I don't f*king know why
/*void solve_jacobi(mesh_t *A, const mesh_t *B, mesh_t *C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    // NOTE: We store pow in an array
    double powers[STENCIL_ORDER];
    for (int o = 1; o <= STENCIL_ORDER; ++o) {
        powers[o-1] = 1.0 / pow(17.0, o);
    }

    usz i, j, k, o;
    for (i = STENCIL_ORDER; i < A->dim_x - STENCIL_ORDER; ++i) {
        for (j = STENCIL_ORDER; j < A->dim_y - STENCIL_ORDER; ++j) {
            for (k = STENCIL_ORDER; k < A->dim_z - STENCIL_ORDER; k += 4) {
                __m256d acc = _mm256_setzero_pd();

                // NOTE: We load A & B for C = A * B (can broadcast_sd maybe ? or load_pd because out of inside loop, memory is alligned ?)
                __m256d self_vals_A = _mm256_loadu_pd(&A->values[i][j][k]);
                __m256d self_vals_B = _mm256_loadu_pd(&B->values[i][j][k]);
                __m256d self_product = _mm256_mul_pd(self_vals_A, self_vals_B);
                acc = _mm256_add_pd(acc, self_product);

                for (o = 1; o <= STENCIL_ORDER; ++o) {
                    double factor = powers[o-1];
                    __m256d factor_vec = _mm256_set1_pd(factor);

                    // Load each pair of neighbors
                    __m256d neighbors_hv_A = _mm256_set_pd(
                        A->values[i+o][j][k], A->values[i-o][j][k],
                        A->values[i][j+o][k], A->values[i][j-o][k]
                    );
                    __m256d neighbors_hv_B = _mm256_set_pd(
                        B->values[i+o][j][k], B->values[i-o][j][k],
                        B->values[i][j+o][k], B->values[i][j-o][k]
                    );

                    __m256d neighbors_k_A = _mm256_set_pd(
                        A->values[i][j][k+o], A->values[i][j][k-o],
                        0.0, 0.0    //NOTE: We just need k + o & k - o, so the last two values of our vector are 0.0
                    );
                    __m256d neighbors_k_B = _mm256_set_pd(
                        B->values[i][j][k+o], B->values[i][j][k-o],
                        0.0, 0.0
                    );

                    // NOTE: A * B + A1 * B1
                    __m256d neighbor_hv_products = _mm256_mul_pd(neighbors_hv_A, neighbors_hv_B);
                    __m256d neighbor_k_products = _mm256_mul_pd(neighbors_k_A, neighbors_k_B);

                    __m256d sum_vec = _mm256_add_pd(neighbor_hv_products, neighbor_k_products);
                    sum_vec = _mm256_mul_pd(sum_vec, factor_vec);

                    acc = _mm256_add_pd(acc, sum_vec);
                }

                // NOTE: We store the acc in C
                _mm256_storeu_pd(&C->values[i][j][k], acc);
            }
        }
    }
    mesh_copy_core(A, C);
}*/



//NOTE: Wrong computation, need to Hadd_pd as said in the next version

/*
void solve_jacobi(mesh_t *A, const mesh_t *B, mesh_t *C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);


    double powers[STENCIL_ORDER];
    for (int o = 1; o <= STENCIL_ORDER; ++o) {
        powers[o-1] = 1.0 / pow(17.0, o);
    }

    usz i, j, k, o;
    for (i = STENCIL_ORDER; i < A->dim_x - STENCIL_ORDER; ++i) {
        for (j = STENCIL_ORDER; j < A->dim_y - STENCIL_ORDER; ++j) {
            for (k = STENCIL_ORDER; k < A->dim_z - STENCIL_ORDER; k += 4) { 
                __m256d acc = _mm256_setzero_pd();

                __m256d self_vals_A = _mm256_loadu_pd(&A->values[i][j][k]);
                __m256d self_vals_B = _mm256_loadu_pd(&B->values[i][j][k]);
                __m256d self_product = _mm256_mul_pd(self_vals_A, self_vals_B);
                acc = _mm256_add_pd(acc, self_product);

                for (o = 1; o <= STENCIL_ORDER; ++o) {
                    double factor = powers[o-1];
                    __m256d factor_vec = _mm256_set1_pd(factor);

                    __m256d neighbors_A = _mm256_set_pd(
                        A->values[i+o][j][k], A->values[i-o][j][k],
                        A->values[i][j+o][k], A->values[i][j-o][k]
                    );
                    __m256d neighbors_B = _mm256_set_pd(
                        B->values[i+o][j][k], B->values[i-o][j][k],
                        B->values[i][j+o][k], B->values[i][j-o][k]
                    );

                    __m256d neighbors_k_A = _mm256_set_pd(
                        A->values[i][j][k+o], A->values[i][j][k-o],
                        0, 0 
                    );
                    __m256d neighbors_k_B = _mm256_set_pd(
                        B->values[i][j][k+o], B->values[i][j][k-o],
                        0, 0
                    );
                    __m256d neighbor_products = _mm256_mul_pd(_mm256_add_pd(neighbors_A, neighbors_k_A), _mm256_add_pd(neighbors_B, neighbors_k_B));
                    __m256d contributions = _mm256_mul_pd(neighbor_products, factor_vec);
                    acc = _mm256_add_pd(acc, contributions);
                }
                _mm256_storeu_pd(&C->values[i][j][k], acc);
            }
        }
    }
    mesh_copy_core(A, C);
}
*/



// NOTE: Maybe I need to try to hadd, better precision ?
/*
void solve_jacobi(mesh_t *A, const mesh_t *B, mesh_t *C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    double powers[STENCIL_ORDER];
    for (int o = 1; o <= STENCIL_ORDER; ++o) {
        powers[o-1] = 1.0 / pow(17.0, o);
    }

    usz i, j, k, o;

    for (i = STENCIL_ORDER; i < A->dim_x - STENCIL_ORDER; ++i) {
        for (j = STENCIL_ORDER; j < A->dim_y - STENCIL_ORDER; ++j) {
            for (k = STENCIL_ORDER; k < A->dim_z - STENCIL_ORDER; ++k) {
                //__m256d acc = _mm256_setzero_pd();

                __m256d self_val_A = _mm256_broadcast_sd(&A->values[i][j][k]);
                __m256d self_val_B = _mm256_broadcast_sd(&B->values[i][j][k]);
                __m256d product = _mm256_mul_pd(self_val_A, self_val_B);
                //acc = _mm256_add_pd(acc, product);

                for (o = 1; o <= STENCIL_ORDER; ++o) {
                    double factor = powers[o-1];

                    __m256d A1 = _mm256_set_pd(
                        A->values[i+o][j][k], A->values[i-o][j][k],
                        0.0, 0.0
                    );
                    __m256d B1 = _mm256_set_pd(
                        B->values[i+o][j][k], B->values[i-o][j][k],
                        0.0, 0.0
                    );

                    __m256d A2 = _mm256_set_pd(
                        A->values[i][j + o][k], A->values[i][j - o][k],
                        0.0, 0.0
                    );
                    __m256d B2 = _mm256_set_pd(
                        B->values[i][j + o][k], B->values[i][j - o][k],
                        0.0, 0.0
                    );

                    __m256d A3 = _mm256_set_pd(
                        A->values[i][j][k + o], A->values[i][j][k - o],
                        0.0, 0.0
                    );
                    __m256d B3 = _mm256_set_pd(
                        B->values[i][j][k + o], B->values[i][j][k - o],
                        0.0, 0.0
                    );
                    // NOTE: A1*B1 & A2*B2 & A3*B3
                    __m256d mul1 = _mm256_mul_pd(A1, B1);
                    __m256d mul2 = _mm256_mul_pd(A2, B2);
                    __m256d mul3 = _mm256_mul_pd(A3, B3);

                    __m256d factor_vec = _mm256_broadcast_sd(&factor);

                    mul1 = _mm256_mul_pd(mul1, factor_vec);
                    mul2 = _mm256_mul_pd(mul2, factor_vec);
                    mul3 = _mm256_mul_pd(mul3, factor_vec); 

/*
                    // NOTE: We can HADD this way if we store each neighbour in the same 
                    // vector for each low and high 128-bits separately, 
                    // We never crossing the 128-bit boundary
                    // Thus doing it that way : compute 4 two-element horizontal sums:
                    // lower 64 bits contain A1[0] + A1[1]
                    // next 64 bits contain B1[0] + B1[1]
                    // next 64 bits contain A1[2] + A1[3]
                    // next 64 bits contain B1[2] + B1[3]
*/
/*
                    __m256 sum = _mm256_hadd_pd(mul1, mul2);
                    __m128d sum_high = _mm256_extractf128_pd(sum, 1);
                    __m128d result = _mm_add_pd(sum_high, _mm256_castpd256_pd128(sum));

                    __m256 sum_2 = _mm256_hadd_pd(result, mul3);
                    __m128d sum_high_2 = _mm256_extractf128_pd(sum_2, 1);
                    __m128d total = _mm_add_pd(sum_high_2, _mm256_castpd256_pd128(sum_2));
*/

/*                    __m256d sum1 = _mm256_add_pd(mul1, mul2);
                    __m256d total_sum = _mm256_add_pd(sum1, mul3);

                    //__m256d contributions = _mm256_mul_pd(total_sum, factor_vec); //faire mul1 * fac_vec etc avant de passer au somme
                    product = _mm256_add_pd(product, total_sum);
                }
                f64 tmp[4];
                _mm256_storeu_pd(tmp, product);
                C->values[i][j][k] = tmp[3] ;
                //C->values[i][j][k] = result;
            }
        }
    }
    mesh_copy_core(A, C);
}*/

void solve_jacobi(mesh_t *A, const mesh_t *B, mesh_t *C, double *powers) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);
/*
    double powers[STENCIL_ORDER];
    for (int o = 1; o <= STENCIL_ORDER; ++o) {
        powers[o-1] = 1.0 / pow(17.0, o);
    }
*/
    for (usz i = STENCIL_ORDER; i < A->dim_x - STENCIL_ORDER; ++i) {
        for (usz j = STENCIL_ORDER; j < A->dim_y - STENCIL_ORDER; ++j) {
            for (usz k = STENCIL_ORDER; k < A->dim_z - STENCIL_ORDER; ++k) {
                __m256d acc = _mm256_setzero_pd();

                __m256d self_val_A = _mm256_broadcast_sd(&A->values[i][j][k]);
                __m256d self_val_B = _mm256_broadcast_sd(&B->values[i][j][k]);
                __m256d product = _mm256_mul_pd(self_val_A, self_val_B);

                acc = _mm256_add_pd(acc, product);

                for (int o = 1; o <= STENCIL_ORDER; ++o) {
                    double factor = powers[o-1];

                    // NOTE: For each neighbor direction (+x, -x, +y, -y, +z, -z)
                    __m256d add_contributions = _mm256_setzero_pd();
                    for (int d = -1; d <= 1; d += 2) {  // NOTE: d takes values -1, +1
                        // X neighbors
                        __m256d A_x = _mm256_broadcast_sd(&A->values[i+d*o][j][k]);
                        __m256d B_x = _mm256_broadcast_sd(&B->values[i+d*o][j][k]);
                        add_contributions = _mm256_fmadd_pd(A_x, B_x, add_contributions);

                        // Y neighbors
                        __m256d A_y = _mm256_broadcast_sd(&A->values[i][j+d*o][k]);
                        __m256d B_y = _mm256_broadcast_sd(&B->values[i][j+d*o][k]);
                        add_contributions = _mm256_fmadd_pd(A_y, B_y, add_contributions);

                        // Z neighbors
                        __m256d A_z = _mm256_broadcast_sd(&A->values[i][j][k+d*o]);
                        __m256d B_z = _mm256_broadcast_sd(&B->values[i][j][k+d*o]);
                        add_contributions = _mm256_fmadd_pd(A_z, B_z, add_contributions);
                    }

                    // NOTE: Multiply by factor and sum to accumulator
                    add_contributions = _mm256_mul_pd(add_contributions, _mm256_broadcast_sd(&factor));
                    acc = _mm256_add_pd(acc, add_contributions);
                }

                // Reduce vector sum to a single sum and store in C
                double partial_results[4];
                _mm256_storeu_pd(partial_results, acc);
                C->values[i][j][k] = /*partial_results[0] + partial_results[1] + partial_results[2] +*/ partial_results[3];
            }
        }
    }

    mesh_copy_core(A, C);
}

