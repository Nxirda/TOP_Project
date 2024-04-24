#include "stencil/solve.h"

#include <assert.h>
#include <math.h>

void solve_jacobi(mesh_t* A, const mesh_t* B, mesh_t* C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    usz const dim_x = A->dim_x;
    usz const dim_y = A->dim_y;
    usz const dim_z = A->dim_z;

    for (usz k = STENCIL_ORDER; k < dim_z - STENCIL_ORDER; ++k) {
        for (usz j = STENCIL_ORDER; j < dim_y - STENCIL_ORDER; ++j) {
            for (usz i = STENCIL_ORDER; i < dim_x - STENCIL_ORDER; ++i) {
                usz index = i * (dim_y * dim_z) + j * dim_z + k;

                // Start by setting the core computation
                C->values[index] = A->values[index] * B->values[index];

                // Apply the Jacobi stencil operation
                for (usz o = 1; o <= STENCIL_ORDER; ++o) {
                    usz indices[6] = {
                        (i + o) * (dim_y * dim_z) + j * dim_z + k,
                        (i - o) * (dim_y * dim_z) + j * dim_z + k,
                        i * (dim_y * dim_z) + (j + o) * dim_z + k,
                        i * (dim_y * dim_z) + (j - o) * dim_z + k,
                        i * (dim_y * dim_z) + j * dim_z + (k + o),
                        i * (dim_y * dim_z) + j * dim_z + (k - o)
                    };

                    for (int idx = 0; idx < 6; ++idx) {
                        C->values[index] += A->values[indices[idx]] * B->values[indices[idx]] / pow(17.0, (f64)o);
                    }
                }
            }
        }
    }

    mesh_copy_core(A, C);
}
