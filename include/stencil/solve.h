#pragma once

#include "mesh.h"

/* void solve_jacobi (mesh_t *A, mesh_t const *B, mesh_t *C,
                   f64 pow_precomputed[static STENCIL_ORDER]); */

void solve_jacobi_2 (mesh_t *A, mesh_t const *B, mesh_t *C,
                     f64 pow_precomputed[static STENCIL_ORDER]);

void solve_jacobi_3 (mesh_t *A, mesh_t const *B, mesh_t *C,
                     f64 pow_precomputed[STENCIL_ORDER]);

void elementwise_multiply_2 (f64 ***A, f64 ***B, f64 ***C, usz dim_x, usz dim_y, usz dim_z);

