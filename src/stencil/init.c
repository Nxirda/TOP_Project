#include "stencil/init.h"
#include "stencil/comm_handler.h"
#include "stencil/mesh.h"

#include "logging.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

//
static f64
compute_core_pressure (usz i, usz j, usz k)
{
  return sin ((f64)k * cos ((f64)i + 0.311) * cos ((f64)j + 0.817) + 0.613);
}

mesh_t
mesh_new_2 (usz base_dim_x, usz base_dim_y, usz base_dim_z, mesh_kind_t kind,
            comm_handler_t const *comm_handler)
{
  usz const ghost_size = 2 * STENCIL_ORDER;
  usz dim_x = (base_dim_x + ghost_size);
  usz dim_y = (base_dim_y + ghost_size);
  usz dim_z = (base_dim_z + ghost_size);

  f64 ***values = (f64 ***)malloc (dim_x * sizeof (f64 **));
  // A->values = (f64 ***)malloc (A->dim_x * sizeof (f64 **));
  if (NULL == values)
    {
      error ("failed to allocate dimension X of mesh of size %zu bytes",
             dim_x);
    }
#pragma omp /* parallel  */for schedule(static, 8)
  for (usz i = 0; i < dim_x; ++i)
    {
      values[i] = (f64 **)malloc (dim_y * sizeof (f64 *));
      // A->values[i] = (f64 **)malloc (A->dim_y * sizeof (f64 *));
      if (NULL == values[i])
        {
          error ("failed to allocate dimension Y of mesh of size %zu bytes",
                 dim_y);
        }
      for (usz j = 0; j < dim_y; ++j)
        {
          values[i][j] = (f64 *)aligned_alloc (32 , dim_z * sizeof (f64));
          // A->values[i][j] = (f64 *)malloc (A->dim_z * sizeof (f64));
          if (NULL == values[i][j])
            {
              error (
                  "failed to allocate dimension Z of mesh of size %zu bytes",
                  dim_z);
            }
          for (usz k = 0; k < dim_z; ++k)
            {
              switch (kind)
                {
                case MESH_KIND_CONSTANT:
                  values[i][j][k] = compute_core_pressure (
                      comm_handler->coord_x + i, comm_handler->coord_y + j,
                      comm_handler->coord_z + k);
                  break;
                case MESH_KIND_INPUT:
                  if ((i >= STENCIL_ORDER && (i < dim_x - STENCIL_ORDER))
                      && (j >= STENCIL_ORDER && (j < dim_y - STENCIL_ORDER))
                      && (k >= STENCIL_ORDER
                          && (k < dim_z - STENCIL_ORDER)))
                    {
                      values[i][j][k] = 1.0;
                    }
                  else
                    {
                      values[i][j][k] = 0.0;
                    }
                  break;
                case MESH_KIND_OUTPUT:
                  values[i][j][k] = 0.0;
                  break;
                default:
                  __builtin_unreachable ();
                }
            }
        }
    }
  return (mesh_t){
    .dim_x = dim_x,
    .dim_y = dim_y,
    .dim_z = dim_z,
    .values = values,
    /* .mesh_kind = kind, */
  };
}

//
/* static void setup_mesh_cell_kinds(mesh_t* mesh) {
    for (usz i = 0; i < mesh->dim_x; ++i) {
        for (usz j = 0; j < mesh->dim_y; ++j) {
            for (usz k = 0; k < mesh->dim_z; ++k) {
                mesh->cells_kind[i][j][k] = mesh_set_cell_kind(mesh, i, j, k);
            }
        }
    }
} */

//
/* void
init_meshes (mesh_t *A, mesh_t *B, mesh_t *C,
             comm_handler_t const *comm_handler)
{
  assert (A->dim_x == B->dim_x && B->dim_x == C->dim_x
          && C->dim_x == comm_handler->loc_dim_x + STENCIL_ORDER * 2);
  assert (A->dim_y == B->dim_y && B->dim_y == C->dim_y
          && C->dim_y == comm_handler->loc_dim_y + STENCIL_ORDER * 2);
  assert (A->dim_z == B->dim_z && B->dim_z == C->dim_z
          && C->dim_z == comm_handler->loc_dim_z + STENCIL_ORDER * 2); */

  /* setup_mesh_cell_kinds(A);
  setup_mesh_cell_kinds(B);
  setup_mesh_cell_kinds(C);  */

  /*  setup_mesh_cell_values (A, comm_handler);
   setup_mesh_cell_values (B, comm_handler);
   setup_mesh_cell_values (C, comm_handler); */

  /* setup_mesh_cell_values_2 (mesh_t * A, usz base_dim_x, usz base_dim_y,
                            usz base_dim_z, mesh_kind_t kind,
                            comm_handler_t const *comm_handler) */
/* } */
