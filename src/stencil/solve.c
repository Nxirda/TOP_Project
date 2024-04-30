#include "stencil/solve.h"

#include <assert.h>
#include <cblas.h>
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>

#define BLOCK_SIZE 64 // Constant for cache blocking

//
void
elementwise_multiply (mesh_t *A, mesh_t const *B, mesh_t const *C)
{
  usz UNROLL_FACTOR = 16;
  usz lim_z = C->dim_z - (C->dim_z % UNROLL_FACTOR);
  usz j, k;

#pragma omp for schedule(static, 8)
  for (usz i = 0; i < C->dim_x; i++)
    {
      for (j = 0; j < C->dim_y; j++)
        {
          for (k = 0; k < lim_z; k += UNROLL_FACTOR)
            {
              usz _k4_ = k + 4;
              usz _k8_ = k + 8;
              usz _k12_ = k + 12;

              __m256d C_reg = _mm256_load_pd (&C->values[i][j][k]);
              __m256d C4_reg = _mm256_load_pd (&C->values[i][j][_k4_]);

              __m256d B_reg = _mm256_load_pd (&B->values[i][j][k]);
              __m256d B4_reg = _mm256_load_pd (&B->values[i][j][_k4_]);

              _mm256_store_pd (&A->values[i][j][k],
                               _mm256_mul_pd (C_reg, B_reg));
              _mm256_store_pd (&A->values[i][j][_k4_],
                               _mm256_mul_pd (C4_reg, B4_reg));

              __m256d C8_reg = _mm256_load_pd (&C->values[i][j][_k8_]);
              __m256d C12_reg = _mm256_load_pd (&C->values[i][j][_k12_]);

              __m256d B8_reg = _mm256_load_pd (&B->values[i][j][_k8_]);
              __m256d B12_reg = _mm256_load_pd (&B->values[i][j][_k12_]);

              _mm256_store_pd (&A->values[i][j][_k8_],
                               _mm256_mul_pd (C8_reg, B8_reg));
              _mm256_store_pd (&A->values[i][j][_k12_],
                               _mm256_mul_pd (C12_reg, B12_reg));
            }

          for (; k < lim_z; k++)
            {
              __m256d C_reg = _mm256_loadu_pd (&C->values[i][j][k]);
              __m256d B_reg = _mm256_loadu_pd (&B->values[i][j][k]);

              _mm256_storeu_pd (&A->values[i][j][k],
                                _mm256_mul_pd (C_reg, B_reg));
            }
        }
    }
}

//
void
solve_jacobi (mesh_t *A, mesh_t const *B, mesh_t *C,
              f64 pow_precomputed[STENCIL_ORDER])
{
    assert (A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert (A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert (A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    usz const lim_x = C->dim_x - STENCIL_ORDER;
    usz const lim_y = C->dim_y - STENCIL_ORDER;
    usz const lim_z = C->dim_z - STENCIL_ORDER;

    __m256d pow_low = _mm256_load_pd (pow_precomputed);
    __m256d pow_high = _mm256_load_pd (4 + pow_precomputed);

#pragma omp for schedule(static, 8)
    for (usz i = STENCIL_ORDER; i < lim_x; ++i)
      {
        for (usz j = STENCIL_ORDER; j < lim_y; ++j)
          {
            for (usz k = STENCIL_ORDER; k < lim_z; ++k)
              {
                __m256d result;

                // Negative A neighbors on x axis
                __m256d niA5_8 = _mm256_set_pd (
                    A->values[i - 8][j][k], A->values[i - 7][j][k],
                    A->values[i - 6][j][k], A->values[i - 5][j][k]);

                __m256d niA1_4 = _mm256_set_pd (
                    A->values[i - 4][j][k], A->values[i - 3][j][k],
                    A->values[i - 2][j][k], A->values[i - 1][j][k]);

                result = _mm256_mul_pd (niA5_8, pow_high);
                result = _mm256_fmadd_pd (niA1_4, pow_low, result);

                // Positive A neighbors on x axis
                __m256d piA1_4 = _mm256_set_pd (
                    A->values[i + 4][j][k], A->values[i + 3][j][k],
                    A->values[i + 2][j][k], A->values[i + 1][j][k]);

                __m256d piA5_8 = _mm256_set_pd (
                    A->values[i + 8][j][k], A->values[i + 7][j][k],
                    A->values[i + 6][j][k], A->values[i + 5][j][k]);

                result = _mm256_fmadd_pd (piA1_4, pow_low, result);
                result = _mm256_fmadd_pd (piA5_8, pow_high, result);

                // Negative A neighbors on y axis
                __m256d njA5_8 = _mm256_set_pd (
                    A->values[i][j - 8][k], A->values[i][j - 7][k],
                    A->values[i][j - 6][k], A->values[i][j - 5][k]);

                __m256d njA1_4 = _mm256_set_pd (
                    A->values[i][j - 4][k], A->values[i][j - 3][k],
                    A->values[i][j - 2][k], A->values[i][j - 1][k]);

                result = _mm256_fmadd_pd (njA5_8, pow_high, result);
                result = _mm256_fmadd_pd (njA1_4, pow_low, result);

                // Positive A neighbors on y axis
                __m256d pjA1_4 = _mm256_set_pd (
                    A->values[i][j + 4][k], A->values[i][j + 3][k],
                    A->values[i][j + 2][k], A->values[i][j + 1][k]);

                __m256d pjA5_8 = _mm256_set_pd (
                    A->values[i][j + 8][k], A->values[i][j + 7][k],
                    A->values[i][j + 6][k], A->values[i][j + 5][k]);

                result = _mm256_fmadd_pd (pjA1_4, pow_low, result);
                result = _mm256_fmadd_pd (pjA5_8, pow_high, result);

                // Negative A neighbors on z axis
                __m256d nkA5_8 = _mm256_set_pd (
                    A->values[i][j][k - 8], A->values[i][j][k - 7],
                    A->values[i][j][k - 6], A->values[i][j][k - 5]);

                __m256d nkA1_4 = _mm256_set_pd (
                    A->values[i][j][k - 4], A->values[i][j][k - 3],
                    A->values[i][j][k - 2], A->values[i][j][k - 1]);

                result = _mm256_fmadd_pd (nkA5_8, pow_high, result);
                result = _mm256_fmadd_pd (nkA1_4, pow_low, result);

                // Positive A neighbors on z axis
                __m256d pkA1_4 = _mm256_set_pd (
                    A->values[i][j][k + 4], A->values[i][j][k + 3],
                    A->values[i][j][k + 2], A->values[i][j][k + 1]);

                __m256d pkA5_8 = _mm256_set_pd (
                    A->values[i][j][k + 8], A->values[i][j][k + 7],
                    A->values[i][j][k + 6], A->values[i][j][k + 5]);

                result = _mm256_fmadd_pd (pkA1_4, pow_low, result);
                result = _mm256_fmadd_pd (pkA5_8, pow_high, result);

                __m128d low = _mm256_extractf128_pd (result, 0);
                __m128d high = _mm256_extractf128_pd (result, 1);

                __m128d sum = _mm_add_pd (high, low);
                sum = _mm_hadd_pd (sum, sum);

                C->values[i][j][k] = A->values[i][j][k] + _mm_cvtsd_f64 (sum);
              }
          }
      }
}