#pragma once

#include "mesh.h"

void solve_jacobi (mesh_t *A, mesh_t *C,
                   f64 pow_precomputed[STENCIL_ORDER],usz BLOCK_SIZE_X, usz BLOCK_SIZE_Y, usz BLOCK_SIZE_Z);

void elementwise_multiply (mesh_t *A, mesh_t const *B, mesh_t const *C);
