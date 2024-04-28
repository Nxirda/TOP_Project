#pragma once

#include "comm_handler.h"
#include "mesh.h"

/* void init_meshes (mesh_t *A, mesh_t *B, mesh_t *C,
                  comm_handler_t const *comm_handler); */

mesh_t
mesh_new_2 (usz base_dim_x, usz base_dim_y, usz base_dim_z, mesh_kind_t kind,
            comm_handler_t const *comm_handler);