#include "stencil/solve.h"

#include <assert.h>
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>

#define BLOCK_SIZE 64  // Constant for cache blocking
#define YMM_D_SIZE 4   // Number of doubles a ymm can store
#define CELL_NEIGH 6   // Number of neighbors a cell has

// NOTE : Passed the func to be layout right (base was layout  left)
void
solve_jacobi (mesh_t *A, mesh_t const *B, mesh_t *C)
{
  assert (A->dim_x == B->dim_x && B->dim_x == C->dim_x);
  assert (A->dim_y == B->dim_y && B->dim_y == C->dim_y);
  assert (A->dim_z == B->dim_z && B->dim_z == C->dim_z);

  // NOTE : Unrolled && Precomputed
  f64 pow_precomputed[STENCIL_ORDER];
  for (usz o = 1; o <= STENCIL_ORDER; o += 4)
    {
      *(pow_precomputed + (o - 1)) = (1.0 / pow (17.0, (f64)(o)));
      *(pow_precomputed + (o + 1 - 1)) = (1.0 / pow (17.0, (f64)(o + 1)));
      *(pow_precomputed + (o + 2 - 1)) = (1.0 / pow (17.0, (f64)(o + 2)));
      *(pow_precomputed + (o + 3 - 1)) = (1.0 / pow (17.0, (f64)(o + 3)));
    }

  usz const lim_x = A->dim_x - STENCIL_ORDER;
  usz const lim_y = A->dim_y - STENCIL_ORDER;
  usz const lim_z = A->dim_z - STENCIL_ORDER;

  // Arrays used for vectorization in the loop
  f64 *restrict temp_ymm_one
      = (f64 *)aligned_alloc (32, YMM_D_SIZE * sizeof (f64));
  // Holds the 2 last elements of the inner loop, 6 values at each iteration
  f64 *restrict temp_ymm_two
      = (f64 *)aligned_alloc (32, (CELL_NEIGH - YMM_D_SIZE) * sizeof (f64));

  // NOTE : Cache blocking
  for (usz x = STENCIL_ORDER; x < lim_x; x += BLOCK_SIZE)
    {
      for (usz y = STENCIL_ORDER; y < lim_y; y += BLOCK_SIZE)
        {
          for (usz z = STENCIL_ORDER; z < lim_z; z += BLOCK_SIZE)
            {
              for (usz i = x; i < x + BLOCK_SIZE && i < lim_x; ++i)
                {
                  for (usz j = y; j < y + BLOCK_SIZE && j < lim_y; ++j)
                    {
                      for (usz k = z; k < z + BLOCK_SIZE && k < lim_z; ++k)
                        {
                          // NOTE : total of 16 YMM reg
                          // Load in ymm reg
                          f64 current_value = A->values[i][j][k]
                                              * B->values[i][j][k];
                          /*
                          NOTE : Factored the power computation
                                  Removed unnecessary memory accesses for
                          C->cells[i][j][k]
                          */
                          for (usz o = 1; o <= STENCIL_ORDER; ++o)
                            {
                              // Load in ymm
                              f64 pow_result = *(pow_precomputed + (o - 1));
                              __m256d ymm_pow
                                  = _mm256_broadcast_sd (&pow_result);

                              *(temp_ymm_one) = A->values[i + o][j][k]
                                                * B->values[i + o][j][k];
                              *(temp_ymm_one + 1)
                                  = A->values[i - o][j][k]
                                    * B->values[i - o][j][k];
                              *(temp_ymm_one + 2)
                                  = A->values[i][j + o][k]
                                    * B->values[i][j + o][k];
                              *(temp_ymm_one + 3)
                                  = A->values[i][j - o][k]
                                    * B->values[i][j - o][k];
                              *(temp_ymm_two) = A->values[i][j][k + o]
                                                * B->values[i][j][k + o];
                              *(temp_ymm_two + 1)
                                  = A->values[i][j][k - o]
                                    * B->values[i][j][k - o];

                              __m256d ymm_one = _mm256_load_pd (temp_ymm_one);
                              __m256d ymm_two
                                  = _mm256_set_pd (0.0, 0.0, *(temp_ymm_two),
                                                   *(temp_ymm_two + 1));

                              __m256d pow_result_1
                                  = _mm256_mul_pd (ymm_one, ymm_pow);
                              __m256d pow_result_2
                                  = _mm256_mul_pd (ymm_two, ymm_pow);

                              __m256d final_res = _mm256_hadd_pd (pow_result_1,
                                                        pow_result_2);
                              _mm256_store_pd (temp_ymm_one, final_res);
                              current_value += *(temp_ymm_one)
                                               + *(temp_ymm_one + 1)
                                               + *(temp_ymm_one + 2)
                                               + *(temp_ymm_one + 3);
                            }
                          C->values[i][j][k] = current_value;
                        }
                    }
                }
            } // END CACHE BLOCK z
        }     // END CACHE BLOCK y
    }         // END CACHE BLOCK x

  free (temp_ymm_one);
  free (temp_ymm_two);
  mesh_copy_core (A, C);
}

// UNUSED VERSION
void solve_jacobi_2(mesh_t *A, mesh_t const *B, mesh_t *C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    usz i, j, k, ii, jj, kk, o;
    double divisor, sum;

    // Précalculer les diviseurs pour réduire les coûts de calcul
    double divisors[STENCIL_ORDER + 1];
    for (o = 1; o <= STENCIL_ORDER; ++o) {
        divisors[o] = pow(17.0, o);
    }

    // Boucles principales avec cache blocking
    for (k = STENCIL_ORDER; k < A->dim_z - STENCIL_ORDER; k += BLOCK_SIZE) {
        for (j = STENCIL_ORDER; j < A->dim_y - STENCIL_ORDER; j += BLOCK_SIZE) {
            for (i = STENCIL_ORDER; i < A->dim_x - STENCIL_ORDER; i += BLOCK_SIZE) {
                // Traitement des blocs
                for (kk = k; kk < k + BLOCK_SIZE && kk < A->dim_z - STENCIL_ORDER; ++kk) {
                    for (jj = j; jj < j + BLOCK_SIZE && jj < A->dim_y - STENCIL_ORDER; ++jj) {
                        for (ii = i; ii < i + BLOCK_SIZE && ii < A->dim_x - STENCIL_ORDER; ++ii) {
                            sum = A->values[ii][jj][kk] * B->values[ii][jj][kk];  // Multiplication de la cellule centrale
                            for (o = 1; o <= STENCIL_ORDER; ++o) {
                                divisor = divisors[o];
                                // Accumulation des sommes pondérées des voisins
                                sum += (A->values[ii + o][jj][kk] * B->values[ii + o][jj][kk] +
                                        A->values[ii - o][jj][kk] * B->values[ii - o][jj][kk] +
                                        A->values[ii][jj + o][kk] * B->values[ii][jj + o][kk] +
                                        A->values[ii][jj - o][kk] * B->values[ii][jj - o][kk] +
                                        A->values[ii][jj][kk + o] * B->values[ii][jj][kk + o] +
                                        A->values[ii][jj][kk - o] * B->values[ii][jj][kk - o]) / divisor;
                            }
                            C->values[ii][jj][kk] = sum;
                        }
                    }
                }
            }
        }
    }

    mesh_copy_core(A, C);
}
/*current_value += A->cells.value[i + o][j][k] *
                                                           B->cells.value[i +
   o][j][k] * (pow_result); current_value += A->cells.value[i - o][j][k] *
                                                           B->cells.value[i -
   o][j][k] * (pow_result); current_value += A->cells.value[i][j + o][k] *
                                                           B->cells.value[i][j
   + o][k] * (pow_result); current_value += A->cells.value[i][j - o][k] *
                                                           B->cells.value[i][j
   - o][k] * (pow_result); current_value += A->cells.value[i][j][k + o] *
                                                           B->cells.value[i][j][k
   + o] * (pow_result); current_value += A->cells.value[i][j][k - o] *
                                                           B->cells.value[i][j][k
   - o] * (pow_result); */

/*for (usz i = STENCIL_ORDER; i < dim_x - STENCIL_ORDER; ++i) {
    for (usz j = STENCIL_ORDER; j < dim_y - STENCIL_ORDER; ++j) {
        for (usz k = STENCIL_ORDER; k < dim_z - STENCIL_ORDER; ++k) {
            f64 current_value = A->cells.value[i][j][k] *
B->cells.value[i][j][k];

            for (usz o = 1; o <= STENCIL_ORDER; ++o) {
                f64 pow_result = *(pow_precomputed + (o-1));
                current_value += A->cells.value[i + o][j][k] *
                                           B->cells.value[i + o][j][k] *
(pow_result); current_value += A->cells.value[i - o][j][k] * B->cells.value[i -
o][j][k] * (pow_result); current_value += A->cells.value[i][j + o][k] *
                                           B->cells.value[i][j + o][k] *
(pow_result); current_value += A->cells.value[i][j - o][k] *
                                           B->cells.value[i][j - o][k] *
(pow_result); current_value += A->cells.value[i][j][k + o] *
                                           B->cells.value[i][j][k + o] *
(pow_result); current_value += A->cells.value[i][j][k - o] *
                                           B->cells.value[i][j][k - o] *
(pow_result);
            }
            C->cells.value[i][j][k] = current_value;
        }
    }
}*/
