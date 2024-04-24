#include "stencil/init.h"

#include "stencil/comm_handler.h"
#include "stencil/mesh.h"

#include <assert.h>
#include <math.h>

static f64 compute_core_pressure(usz i, usz j, usz k) {
    return sin((f64)k * cos((f64)i + 0.311) * cos((f64)j + 0.817) + 0.613);
}

static void setup_mesh_cell_values(mesh_t* mesh, comm_handler_t const* comm_handler) {
    for (usz i = 0; i < mesh->dim_x; ++i) {
        for (usz j = 0; j < mesh->dim_y; ++j) {
            for (usz k = 0; k < mesh->dim_z; ++k) {
                usz index = i * (mesh->dim_y * mesh->dim_z) + j * mesh->dim_z + k;
                f64 value;
                switch (mesh->kind) {
                    case MESH_KIND_CONSTANT:
                        value = compute_core_pressure(
                            comm_handler->coord_x + i,
                            comm_handler->coord_y + j,
                            comm_handler->coord_z + k
                        );
                        break;
                    case MESH_KIND_INPUT:
                        if ((i >= STENCIL_ORDER && i < mesh->dim_x - STENCIL_ORDER) &&
                            (j >= STENCIL_ORDER && j < mesh->dim_y - STENCIL_ORDER) &&
                            (k >= STENCIL_ORDER && k < mesh->dim_z - STENCIL_ORDER)) {
                            value = 1.0;
                        } else {
                            value = 0.0;
                        }
                        break;
                    case MESH_KIND_OUTPUT:
                        value = 0.0;
                        break;
                    default:
                        assert(0); // Unreachable code
                }
                mesh->values[index] = value;
            }
        }
    }
}


static void setup_mesh_cell_kinds(mesh_t* mesh) {
    for (usz i = 0; i < mesh->dim_x; ++i) {
        for (usz j = 0; j < mesh->dim_y; ++j) {
            for (usz k = 0; k < mesh->dim_z; ++k) {
                usz index = i * (mesh->dim_y * mesh->dim_z) + j * mesh->dim_z + k;
                mesh->kinds[index] = mesh_set_cell_kind(mesh, i, j, k);
            }
        }
    }
}


void init_meshes(mesh_t* A, mesh_t* B, mesh_t* C, comm_handler_t const* comm_handler) {
    assert(A && B && C && "Mesh pointers must not be NULL");
    assert(
        A->dim_x == B->dim_x && B->dim_x == C->dim_x &&
        C->dim_x == comm_handler->loc_dim_x + STENCIL_ORDER * 2 &&
        A->dim_y == B->dim_y && B->dim_y == C->dim_y &&
        C->dim_y == comm_handler->loc_dim_y + STENCIL_ORDER * 2 &&
        A->dim_z == B->dim_z && B->dim_z == C->dim_z &&
        C->dim_z == comm_handler->loc_dim_z + STENCIL_ORDER * 2
    );

    setup_mesh_cell_kinds(A);
    setup_mesh_cell_kinds(B);
    setup_mesh_cell_kinds(C);

    setup_mesh_cell_values(A, comm_handler);
    setup_mesh_cell_values(B, comm_handler);
    setup_mesh_cell_values(C, comm_handler);
}
