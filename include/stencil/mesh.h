#pragma once

#include "types.h"

#define STENCIL_ORDER 8UL

//
typedef enum mesh_kind_e
{
  MESH_KIND_CONSTANT,
  MESH_KIND_INPUT,
  MESH_KIND_OUTPUT,
} __attribute__ ((packed)) mesh_kind_t;

/// Three-dimensional mesh.
/// Storage of cells is in layout right (aka RowMajor)
typedef struct mesh_s
{
  f64 ***values;
  usz dim_x;
  usz dim_y;
  usz dim_z;
} mesh_t;

/// De-initialize a mesh.
void mesh_drop (mesh_t *self);