#include "stencil/solve.h"

#include <assert.h>
#include <cblas.h>
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>

#define BLOCK_SIZE 64 // Constant for cache blocking
#define YMM_D_SIZE 4  // Number of doubles a ymm can store
#define CELL_NEIGH 6  // Number of neighbors a cell has

/* //
void
elementwise_multiply (f64 ***A, f64 ***B, usz dim_x, usz dim_y, usz dim_z)
{
  usz i, j, k;
  usz UNROLL_FACTOR = 16;
  usz lim = dim_z - (dim_z % UNROLL_FACTOR);

  for (i = STENCIL_ORDER; i < dim_x; i++)
    {
      for (j = STENCIL_ORDER; j < dim_y; j++)
        {
          for (k = STENCIL_ORDER; k < lim; k += UNROLL_FACTOR)
            {
              usz _k4_ = k + 4;
              usz _k8_ = k + 8;
              usz _k12_ = k + 12;

              __m256d A_reg = _mm256_loadu_pd (&A[i][j][k]);
              __m256d B_reg = _mm256_loadu_pd (&B[i][j][k]);

              _mm256_storeu_pd (&A[i][j][k], _mm256_mul_pd (A_reg, B_reg));

              __m256d A4_reg = _mm256_loadu_pd (&A[i][j][_k4_]);
              __m256d B4_reg = _mm256_loadu_pd (&B[i][j][_k4_]);

              _mm256_storeu_pd (&A[i][j][_k4_],
                                _mm256_mul_pd (A4_reg, B4_reg));

              __m256d A8_reg = _mm256_loadu_pd (&A[i][j][_k8_]);
              __m256d B8_reg = _mm256_loadu_pd (&B[i][j][_k8_]);

              _mm256_storeu_pd (&A[i][j][_k8_],
                                _mm256_mul_pd (A8_reg, B8_reg));

              __m256d A12_reg = _mm256_loadu_pd (&A[i][j][_k12_]);
              __m256d B12_reg = _mm256_loadu_pd (&B[i][j][_k12_]);


              _mm256_storeu_pd (&A[i][j][_k12_],
                                _mm256_mul_pd (A12_reg, B12_reg));
            }

          for (k = lim; k < dim_z; k++)
            {
              __m256d A_reg = _mm256_loadu_pd (&A[i][j][k]);
              __m256d B_reg = _mm256_loadu_pd (&B[i][j][k]);

              _mm256_storeu_pd (&A[i][j][k], _mm256_mul_pd (A_reg, B_reg));
            }
        }
    }
} */

// NOTE : Passed the func to be layout right (base was layout  left)
/*void
solve_jacobi (mesh_t *A, mesh_t const *B, mesh_t *C,
              f64 pow_precomputed[static STENCIL_ORDER])
{
  assert (A->dim_x == B->dim_x && B->dim_x == C->dim_x);
  assert (A->dim_y == B->dim_y && B->dim_y == C->dim_y);
  assert (A->dim_z == B->dim_z && B->dim_z == C->dim_z);

  usz const lim_x = A->dim_x - STENCIL_ORDER;
  usz const lim_y = A->dim_y - STENCIL_ORDER;
  usz const lim_z = A->dim_z - STENCIL_ORDER;

  // Arrays used for vectorization in the loop
  f64 *restrict temp_ymm_one
      = (f64 *)aligned_alloc (32, YMM_D_SIZE * sizeof (f64));
  // Holds the 2 last elements of the inner loop, 6 values at each iteration
  f64 *restrict temp_ymm_two
      = (f64 *)aligned_alloc (32, (CELL_NEIGH - YMM_D_SIZE) * sizeof (f64));

  elementwise_multiply (A->values, B->values, lim_x, lim_y, lim_z);

  //  NOTE : Cache blocking
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
                          f64 current_value = A->values[i][j][k];

                          NOTE : Factored the power computation
                                  Removed unnecessary memory accesses for
                          C->cells[i][j][k]

                          for (usz o = 1; o <= STENCIL_ORDER; ++o)
                            {
                              // Load in ymm
                              f64 pow_result = *(pow_precomputed + (o - 1));
                              __m256d ymm_pow
                                  = _mm256_broadcast_sd (&pow_result);

                              *(temp_ymm_one) = A->values[i + o][j][k];
                              *(temp_ymm_one + 1) = A->values[i - o][j][k];
                              *(temp_ymm_one + 2) = A->values[i][j + o][k];
                              *(temp_ymm_one + 3) = A->values[i][j - o][k];

                              *(temp_ymm_two) = A->values[i][j][k + o];
                              *(temp_ymm_two + 1) = A->values[i][j][k - o];

                              __m256d ymm_one = _mm256_load_pd (temp_ymm_one);
                              __m256d ymm_two
                                  = _mm256_set_pd (0.0, 0.0, *(temp_ymm_two),
                                                   *(temp_ymm_two + 1));

                              __m256d pow_result_1
                                  = _mm256_mul_pd (ymm_one, ymm_pow);
                              __m256d pow_result_2
                                  = _mm256_mul_pd (ymm_two, ymm_pow);

                              __m256d final_res = _mm256_hadd_pd (
                                  pow_result_1, pow_result_2);
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
 }*/

// NOTE : Passed the func to be layout right (base was layout  left)
/*void
solve_jacobi_2 (mesh_t *A, mesh_t const *B, mesh_t *C,
                f64 pow_precomputed[static STENCIL_ORDER])
{
  assert (A->dim_x == B->dim_x && B->dim_x == C->dim_x);
  assert (A->dim_y == B->dim_y && B->dim_y == C->dim_y);
  assert (A->dim_z == B->dim_z && B->dim_z == C->dim_z);

  usz const lim_x = A->dim_x - STENCIL_ORDER;
  usz const lim_y = A->dim_y - STENCIL_ORDER;
  usz const lim_z = A->dim_z - STENCIL_ORDER;

  elementwise_multiply (A->values, B->values, lim_x, lim_y, lim_z);

  __m256d pow_low = _mm256_load_pd (pow_precomputed);
  __m256d pow_high = _mm256_load_pd (4 + pow_precomputed);
  f64 temp[4];
  //  NOTE : Cache blocking
   for (usz x = STENCIL_ORDER; x < lim_x; x += BLOCK_SIZE)
    {
      for (usz y = STENCIL_ORDER; y < lim_y; y += BLOCK_SIZE)
        {
          for (usz z = STENCIL_ORDER; z < lim_z; z += BLOCK_SIZE)
            {
  for (usz i = STENCIL_ORDER; i < lim_x; ++i)
    {
      for (usz j = STENCIL_ORDER; j < lim_y; ++j)
        {
          for (usz k = STENCIL_ORDER; k < lim_z; ++k)
            {
              // NOTE : total of 16 YMM reg
              // Load in ymm reg
              // f64 current_value = A->values[i][j][k];

              NOTE : Factored the power computation
                      Removed unnecessary memory accesses for
              C->cells[i][j][k]

              // Positive A neighbors on x axis
              __m256d piA5_8 = _mm256_set_pd (
                  A->values[i + 8][j][k], A->values[i + 7][j][k],
                  A->values[i + 6][j][k], A->values[i + 5][j][k]);

              __m256d piA1_4 = _mm256_set_pd (
                  A->values[i + 4][j][k], A->values[i + 3][j][k],
                  A->values[i + 2][j][k], A->values[i + 1][j][k]);

              // Negative A neighbors on x axis
              __m256d niA1_4 = _mm256_set_pd (
                  A->values[i - 4][j][k], A->values[i - 3][j][k],
                  A->values[i - 2][j][k], A->values[i - 1][j][k]);

              __m256d niA5_8 = _mm256_set_pd (
                  A->values[i - 8][j][k], A->values[i - 7][j][k],
                  A->values[i - 6][j][k], A->values[i - 5][j][k]);

              // Positive A neighbors on y axis
              __m256d pjA5_8 = _mm256_set_pd (
                  A->values[i][j + 8][k], A->values[i][j + 7][k],
                  A->values[i][j + 6][k], A->values[i][j + 5][k]);

              __m256d pjA1_4 = _mm256_set_pd (
                  A->values[i][j + 4][k], A->values[i][j + 3][k],
                  A->values[i][j + 2][k], A->values[i][j + 1][k]);

              // Negative A neighbors on y axis
              __m256d njA1_4 = _mm256_set_pd (
                  A->values[i][j - 4][k], A->values[i][j - 3][k],
                  A->values[i][j - 2][k], A->values[i][j - 1][k]);

              __m256d njA5_8 = _mm256_set_pd (
                  A->values[i][j - 8][k], A->values[i][j - 7][k],
                  A->values[i][j - 6][k], A->values[i][j - 5][k]);

              // Positive A neighbors on z axis
              __m256d pkA5_8 = _mm256_set_pd (
                  A->values[i][j][k + 8], A->values[i][j][k + 7],
                  A->values[i][j][k + 6], A->values[i][j][k + 5]);

              __m256d pkA1_4 = _mm256_set_pd (
                  A->values[i][j][k + 4], A->values[i][j][k + 3],
                  A->values[i][j][k + 2], A->values[i][j][k + 1]);

              // Negative A neighbors on z axis
              __m256d nkA1_4 = _mm256_set_pd (
                  A->values[i][j][k - 4], A->values[i][j][k - 3],
                  A->values[i][j][k - 2], A->values[i][j][k - 1]);

              __m256d nkA5_8 = _mm256_set_pd (
                  A->values[i][j][k - 8], A->values[i][j][k - 7],
                  A->values[i][j][k - 6], A->values[i][j][k - 5]);

              // Computations
              __m256d result;
              result = _mm256_mul_pd (piA1_4, pow_low);
              result = _mm256_fmadd_pd (pjA1_4, pow_low, result);
              result = _mm256_fmadd_pd (pkA1_4, pow_low, result);
              result = _mm256_fmadd_pd (niA1_4, pow_low, result);
              result = _mm256_fmadd_pd (njA1_4, pow_low, result);
              result = _mm256_fmadd_pd (nkA1_4, pow_low, result);

              result = _mm256_fmadd_pd (piA5_8, pow_high, result);
              result = _mm256_fmadd_pd (pjA5_8, pow_high, result);
              result = _mm256_fmadd_pd (pkA5_8, pow_high, result);
              result = _mm256_fmadd_pd (niA5_8, pow_high, result);
              result = _mm256_fmadd_pd (njA5_8, pow_high, result);
              result = _mm256_fmadd_pd (nkA5_8, pow_high, result);

              _mm256_store_pd (temp, result);

              C->values[i][j][k] = A->values[i][j][k] + *(temp) + *(temp + 1)
                                   + *(temp + 2) + *(temp + 3);
            }
        }
    }
  //} // END CACHE BLOCK z
    }     // END CACHE BLOCK y
}         // END CACHE BLOCK x

   free (temp_ymm_one);
  free (temp_ymm_two);
   mesh_copy_core (A, C);
 }*/

//
void
elementwise_multiply_2 (f64 ***A, f64 ***B, f64 ***C, usz dim_x, usz dim_y,
                        usz dim_z)
{
  // usz i, j, k;
  usz UNROLL_FACTOR = 16;
  usz lim = dim_z - (dim_z % UNROLL_FACTOR);

#pragma omp for collapse(2)
  for (usz i = STENCIL_ORDER; i < dim_x; i++)
    {
      for (usz j = STENCIL_ORDER; j < dim_y; j++)
        {
          for (usz k = STENCIL_ORDER; k < lim; k += UNROLL_FACTOR)
            {
              usz _k4_ = k + 4;
              usz _k8_ = k + 8;
              usz _k12_ = k + 12;

              __m256d C_reg = _mm256_loadu_pd (&C[i][j][k]);
              __m256d B_reg = _mm256_loadu_pd (&B[i][j][k]);

              __m256d C4_reg = _mm256_loadu_pd (&C[i][j][_k4_]);
              __m256d B4_reg = _mm256_loadu_pd (&B[i][j][_k4_]);

              _mm256_storeu_pd (&A[i][j][k], _mm256_mul_pd (C_reg, B_reg));
              _mm256_storeu_pd (&A[i][j][_k4_],
                                _mm256_mul_pd (C4_reg, B4_reg));

              __m256d C8_reg = _mm256_loadu_pd (&C[i][j][_k8_]);
              __m256d B8_reg = _mm256_loadu_pd (&B[i][j][_k8_]);

              __m256d C12_reg = _mm256_loadu_pd (&C[i][j][_k12_]);
              __m256d B12_reg = _mm256_loadu_pd (&B[i][j][_k12_]);

              _mm256_storeu_pd (&A[i][j][_k8_],
                                _mm256_mul_pd (C8_reg, B8_reg));
              _mm256_storeu_pd (&A[i][j][_k12_],
                                _mm256_mul_pd (C12_reg, B12_reg));
            }

          for (usz k = lim; k < dim_z; k++)
            {
              __m256d C_reg = _mm256_loadu_pd (&C[i][j][k]);
              __m256d B_reg = _mm256_loadu_pd (&B[i][j][k]);

              _mm256_storeu_pd (&A[i][j][k], _mm256_mul_pd (C_reg, B_reg));
            }
        }
    }
}

// NOTE : Passed the func to be layout right (base was layout  left)
// NOTE : removed the copy core
void
solve_jacobi_3 (mesh_t *A, mesh_t const *B, mesh_t *C,
                f64 pow_precomputed[STENCIL_ORDER])
{
#pragma omp parallel
  {
    assert (A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert (A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert (A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    usz const lim_x = C->dim_x - STENCIL_ORDER;
    usz const lim_y = C->dim_y - STENCIL_ORDER;
    usz const lim_z = C->dim_z - STENCIL_ORDER;

    elementwise_multiply_2 (A->values, B->values, C->values, lim_x, lim_y,
                            lim_z);

    __m256d pow_low = _mm256_load_pd (pow_precomputed);
    __m256d pow_high = _mm256_load_pd (4 + pow_precomputed);
    f64 temp[4];
#pragma omp for collapse(2)
    for (usz i = STENCIL_ORDER; i < lim_x; ++i)
      {
        for (usz j = STENCIL_ORDER; j < lim_y; ++j)
          {
            for (usz k = STENCIL_ORDER; k < lim_z; ++k)
              {
                // Positive A neighbors on x axis
                __m256d piA5_8 = _mm256_set_pd (
                    A->values[i + 8][j][k], A->values[i + 7][j][k],
                    A->values[i + 6][j][k], A->values[i + 5][j][k]);

                __m256d piA1_4 = _mm256_set_pd (
                    A->values[i + 4][j][k], A->values[i + 3][j][k],
                    A->values[i + 2][j][k], A->values[i + 1][j][k]);

                // Negative A neighbors on x axis
                __m256d niA1_4 = _mm256_set_pd (
                    A->values[i - 4][j][k], A->values[i - 3][j][k],
                    A->values[i - 2][j][k], A->values[i - 1][j][k]);

                __m256d niA5_8 = _mm256_set_pd (
                    A->values[i - 8][j][k], A->values[i - 7][j][k],
                    A->values[i - 6][j][k], A->values[i - 5][j][k]);

                // Positive A neighbors on y axis
                __m256d pjA5_8 = _mm256_set_pd (
                    A->values[i][j + 8][k], A->values[i][j + 7][k],
                    A->values[i][j + 6][k], A->values[i][j + 5][k]);

                __m256d pjA1_4 = _mm256_set_pd (
                    A->values[i][j + 4][k], A->values[i][j + 3][k],
                    A->values[i][j + 2][k], A->values[i][j + 1][k]);

                // Negative A neighbors on y axis
                __m256d njA1_4 = _mm256_set_pd (
                    A->values[i][j - 4][k], A->values[i][j - 3][k],
                    A->values[i][j - 2][k], A->values[i][j - 1][k]);

                __m256d njA5_8 = _mm256_set_pd (
                    A->values[i][j - 8][k], A->values[i][j - 7][k],
                    A->values[i][j - 6][k], A->values[i][j - 5][k]);

                // Positive A neighbors on z axis
                __m256d pkA5_8 = _mm256_set_pd (
                    A->values[i][j][k + 8], A->values[i][j][k + 7],
                    A->values[i][j][k + 6], A->values[i][j][k + 5]);

                __m256d pkA1_4 = _mm256_set_pd (
                    A->values[i][j][k + 4], A->values[i][j][k + 3],
                    A->values[i][j][k + 2], A->values[i][j][k + 1]);

                // Negative A neighbors on z axis
                __m256d nkA1_4 = _mm256_set_pd (
                    A->values[i][j][k - 4], A->values[i][j][k - 3],
                    A->values[i][j][k - 2], A->values[i][j][k - 1]);

                __m256d nkA5_8 = _mm256_set_pd (
                    A->values[i][j][k - 8], A->values[i][j][k - 7],
                    A->values[i][j][k - 6], A->values[i][j][k - 5]);

                // Computations  REORGANISE IN AN INTELLIGENT WAY
                __m256d result;
                result = _mm256_mul_pd (piA1_4, pow_low);
                result = _mm256_fmadd_pd (pjA1_4, pow_low, result);
                result = _mm256_fmadd_pd (pkA1_4, pow_low, result);
                result = _mm256_fmadd_pd (niA1_4, pow_low, result);
                result = _mm256_fmadd_pd (njA1_4, pow_low, result);
                result = _mm256_fmadd_pd (nkA1_4, pow_low, result);

                result = _mm256_fmadd_pd (piA5_8, pow_high, result);
                result = _mm256_fmadd_pd (pjA5_8, pow_high, result);
                result = _mm256_fmadd_pd (pkA5_8, pow_high, result);
                result = _mm256_fmadd_pd (niA5_8, pow_high, result);
                result = _mm256_fmadd_pd (njA5_8, pow_high, result);
                result = _mm256_fmadd_pd (nkA5_8, pow_high, result);

                _mm256_store_pd (temp, result);

                C->values[i][j][k] = A->values[i][j][k] + *(temp) + *(temp + 1)
                                     + *(temp + 2) + *(temp + 3);
              }
          }
      }
  } // End pragma omp parallel
}