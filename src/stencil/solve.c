#include "stencil/solve.h"

#include <assert.h>
#include <cblas.h>
#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>

/* #define BLOCK_SIZE_X 256 // Constant for cache blocking
#define BLOCK_SIZE_Y 512
#define BLOCK_SIZE_Z 4096 */

#define UNROLL_FACTOR 16



//
void
elementwise_multiply (mesh_t *A, mesh_t const *B, mesh_t const *C)
{
  usz lim_z = C->dim_z - (C->dim_z % UNROLL_FACTOR);
  usz j, k;
  usz dim_x = C->dim_x;
  usz dim_y = C->dim_y;
  usz dim_z = C->dim_z;

  f64 (*A_Matrix)[dim_y][dim_z] = make_3dspan (f64, , A->values, dim_y, dim_z);
  f64 (*B_Matrix)[dim_y][dim_z] = make_3dspan (f64, , B->values, dim_y, dim_z);
  f64 (*C_Matrix)[dim_y][dim_z] = make_3dspan (f64, , C->values, dim_y, dim_z);

#pragma omp for schedule(static, 8)
  for (usz i = 0; i < dim_x; i++)
    {
      for (j = 0; j < dim_y; j++)
        {
          for (k = 0; k < lim_z; k += UNROLL_FACTOR)
            {

              /*  A_Matrix[i][j][k] = C_Matrix[i][j][k] * B_Matrix[i][j][k]; */
              usz _k4_ = k + 4;
              usz _k8_ = k + 8;
              usz _k12_ = k + 12;

              __m256d C_reg = _mm256_load_pd (&C_Matrix[i][j][k]);
              __m256d C4_reg = _mm256_load_pd (&C_Matrix[i][j][_k4_]);

              __m256d B_reg = _mm256_load_pd (&B_Matrix[i][j][k]);
              __m256d B4_reg = _mm256_load_pd (&B_Matrix[i][j][_k4_]);

              _mm256_store_pd (&A_Matrix[i][j][k],
                              _mm256_mul_pd (C_reg, B_reg));
              _mm256_store_pd (&A_Matrix[i][j][_k4_],
                              _mm256_mul_pd (C4_reg, B4_reg));

              __m256d C8_reg = _mm256_load_pd (&C_Matrix[i][j][_k8_]);
              __m256d C12_reg = _mm256_load_pd (&C_Matrix[i][j][_k12_]);

              __m256d B8_reg = _mm256_load_pd (&B_Matrix[i][j][_k8_]);
              __m256d B12_reg = _mm256_load_pd (&B_Matrix[i][j][_k12_]);

              _mm256_store_pd (&A_Matrix[i][j][_k8_],
                              _mm256_mul_pd (C8_reg, B8_reg));
              _mm256_store_pd (&A_Matrix[i][j][_k12_],
                              _mm256_mul_pd (C12_reg, B12_reg));
            }

          for (; k < lim_z; k++)
            {
              __m256d C_reg = _mm256_loadu_pd (&C_Matrix[i][j][k]);
              __m256d B_reg = _mm256_loadu_pd (&B_Matrix[i][j][k]);

              _mm256_storeu_pd (&A_Matrix[i][j][k],
                                _mm256_mul_pd (C_reg, B_reg));
            }
        }
    }
}


//
void
solve_jacobi (mesh_t *A, mesh_t *C,
              f64 pow_precomputed[STENCIL_ORDER], usz BLOCK_SIZE_X, usz BLOCK_SIZE_Y, usz BLOCK_SIZE_Z)
{
  /* assert (A->dim_x == B->dim_x && B->dim_x == C->dim_x);
  assert (A->dim_y == B->dim_y && B->dim_y == C->dim_y);
  assert (A->dim_z == B->dim_z && B->dim_z == C->dim_z); */

  usz const lim_x = C->dim_x - STENCIL_ORDER;
  usz const lim_y = C->dim_y - STENCIL_ORDER;
  usz const lim_z = C->dim_z - STENCIL_ORDER;

  i32 send_buffer_index = 0;

  usz dim_x = C->dim_x;
  usz dim_y = C->dim_y;
  usz dim_z = C->dim_z;

  __m256d pow_low = _mm256_load_pd (pow_precomputed);
  __m256d pow_high = _mm256_load_pd (4 + pow_precomputed);

  f64 (*A_Matrix)[dim_y][dim_z] = make_3dspan (f64, , A->values, dim_y, dim_z);
  f64 (*C_Matrix)[dim_y][dim_z] = make_3dspan (f64, , C->values, dim_y, dim_z);

#pragma omp for schedule(static, 8)
for(usz bx = STENCIL_ORDER; bx < lim_x; bx+= BLOCK_SIZE_X){
  for(usz by = STENCIL_ORDER; by < lim_y; by+= BLOCK_SIZE_Y){
    for(usz bz = STENCIL_ORDER; bz < lim_z; bz+= BLOCK_SIZE_Z){
      usz block_lim_x = (bx + BLOCK_SIZE_X < lim_x) ? bx + BLOCK_SIZE_X : lim_x;
      usz block_lim_y = (by + BLOCK_SIZE_Y < lim_y) ? by + BLOCK_SIZE_Y : lim_y;
      usz block_lim_z = (bz + BLOCK_SIZE_Z < lim_z) ? bz + BLOCK_SIZE_Z : lim_z;

  for (usz i = bx ; i < block_lim_x; ++i)
    {
      for (usz j = by; j < block_lim_y; ++j)
        {
          for (usz k = bz; k < block_lim_z; ++k)
            {
              __m256d result;

              // Negative A neighbors on x axis
              __m256d niA5_8 = _mm256_set_pd (
                  A_Matrix[i - 8][j][k], A_Matrix[i - 7][j][k],
                  A_Matrix[i - 6][j][k], A_Matrix[i - 5][j][k]);

              __m256d niA1_4 = _mm256_set_pd (
                  A_Matrix[i - 4][j][k], A_Matrix[i - 3][j][k],
                  A_Matrix[i - 2][j][k], A_Matrix[i - 1][j][k]);

              result = _mm256_mul_pd (niA5_8, pow_high);
              result = _mm256_fmadd_pd (niA1_4, pow_low, result);

              // Positive A neighbors on x axis
              __m256d piA1_4 = _mm256_set_pd (
                  A_Matrix[i + 4][j][k], A_Matrix[i + 3][j][k],
                  A_Matrix[i + 2][j][k], A_Matrix[i + 1][j][k]);

              __m256d piA5_8 = _mm256_set_pd (
                  A_Matrix[i + 8][j][k], A_Matrix[i + 7][j][k],
                  A_Matrix[i + 6][j][k], A_Matrix[i + 5][j][k]);

              result = _mm256_fmadd_pd (piA1_4, pow_low, result);
              result = _mm256_fmadd_pd (piA5_8, pow_high, result);

              // Negative A neighbors on y axis
              __m256d njA5_8 = _mm256_set_pd (
                  A_Matrix[i][j - 8][k], A_Matrix[i][j - 7][k],
                  A_Matrix[i][j - 6][k], A_Matrix[i][j - 5][k]);

              __m256d njA1_4 = _mm256_set_pd (
                  A_Matrix[i][j - 4][k], A_Matrix[i][j - 3][k],
                  A_Matrix[i][j - 2][k], A_Matrix[i][j - 1][k]);

              result = _mm256_fmadd_pd (njA5_8, pow_high, result);
              result = _mm256_fmadd_pd (njA1_4, pow_low, result);

              // Positive A neighbors on y axis
              __m256d pjA1_4 = _mm256_set_pd (
                  A_Matrix[i][j + 4][k], A_Matrix[i][j + 3][k],
                  A_Matrix[i][j + 2][k], A_Matrix[i][j + 1][k]);

              __m256d pjA5_8 = _mm256_set_pd (
                  A_Matrix[i][j + 8][k], A_Matrix[i][j + 7][k],
                  A_Matrix[i][j + 6][k], A_Matrix[i][j + 5][k]);

              result = _mm256_fmadd_pd (pjA1_4, pow_low, result);
              result = _mm256_fmadd_pd (pjA5_8, pow_high, result);

              // Negative A neighbors on z axis
              __m256d nkA5_8 = _mm256_set_pd (
                  A_Matrix[i][j][k - 8], A_Matrix[i][j][k - 7],
                  A_Matrix[i][j][k - 6], A_Matrix[i][j][k - 5]);

              __m256d nkA1_4 = _mm256_set_pd (
                  A_Matrix[i][j][k - 4], A_Matrix[i][j][k - 3],
                  A_Matrix[i][j][k - 2], A_Matrix[i][j][k - 1]);

              result = _mm256_fmadd_pd (nkA5_8, pow_high, result);
              result = _mm256_fmadd_pd (nkA1_4, pow_low, result);

              // Positive A neighbors on z axis
              __m256d pkA1_4 = _mm256_set_pd (
                  A_Matrix[i][j][k + 4], A_Matrix[i][j][k + 3],
                  A_Matrix[i][j][k + 2], A_Matrix[i][j][k + 1]);

              __m256d pkA5_8 = _mm256_set_pd (
                  A_Matrix[i][j][k + 8], A_Matrix[i][j][k + 7],
                  A_Matrix[i][j][k + 6], A_Matrix[i][j][k + 5]);

              result = _mm256_fmadd_pd (pkA1_4, pow_low, result);
              result = _mm256_fmadd_pd (pkA5_8, pow_high, result);

              __m128d low = _mm256_extractf128_pd (result, 0);
              __m128d high = _mm256_extractf128_pd (result, 1);

              __m128d sum = _mm_add_pd (high, low);
              sum = _mm_hadd_pd (sum, sum);

              C_Matrix[i][j][k] = A_Matrix[i][j][k] + _mm_cvtsd_f64 (sum);
            }
        }
    }
        }
      }
    }
}