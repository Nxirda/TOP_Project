#pragma once

#include "comm_handler.h"
#include "mesh.h"

mesh_t mesh_new (usz base_dim_x, usz base_dim_y, usz base_dim_z,
                 mesh_kind_t kind, comm_handler_t const *comm_handler);