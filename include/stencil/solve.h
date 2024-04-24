#pragma once

#include "mesh.h"

void solve_jacobi (mesh_t *A, mesh_t const *B, mesh_t *C,
                   f64 pow_precomputed[static STENCIL_ORDER]);

void solve_jacobi_2 (mesh_t *A, mesh_t const *B, mesh_t *C);
