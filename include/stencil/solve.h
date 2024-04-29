#pragma once

#include "mesh.h"

void solve_jacobi (mesh_t *A, mesh_t const *B, mesh_t *C,
                   f64 pow_precomputed[STENCIL_ORDER]);

void elementwise_multiply (mesh_t *A, mesh_t const *B, mesh_t const *C);
