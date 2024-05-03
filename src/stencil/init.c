#include "stencil/init.h"
#include "stencil/comm_handler.h"
#include "stencil/mesh.h"

#include "logging.h"
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

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
  usz dim_x = base_dim_x + ghost_size;
  usz dim_y = base_dim_y + ghost_size;
  usz dim_z = base_dim_z + ghost_size;

  // Calculate the size of the 1D array
  usz total_size = dim_x * dim_y * dim_z;

  // Allocate memory for the 1D array
  f64 *values = (f64 *)aligned_alloc (64, total_size * sizeof (f64));
  if (values == NULL)
    {
      error ("failed to allocate mesh of size %zu bytes",
             total_size * sizeof (f64));
    }

  f64 (*casted_values)[dim_y][dim_z] = (f64 (*)[dim_y][dim_z])values;
  // Fill the 1D array with appropriate values
#pragma omp parallel for schedule (static, 8)
  for (usz i = 0; i < dim_x; ++i)
    {
      for (usz j = 0; j < dim_y; ++j)
        {
          for (usz k = 0; k < dim_z; ++k)
            {
              switch (kind)
                {
                case MESH_KIND_CONSTANT:
                  casted_values[i][j][k] = compute_core_pressure (
                      comm_handler->coord_x + i, comm_handler->coord_y + j,
                      comm_handler->coord_z + k);
                  break;
                case MESH_KIND_INPUT:
                  if ((i >= STENCIL_ORDER && (i < dim_x - STENCIL_ORDER))
                      && (j >= STENCIL_ORDER && (j < dim_y - STENCIL_ORDER))
                      && (k >= STENCIL_ORDER && (k < dim_z - STENCIL_ORDER)))
                    {
                      casted_values[i][j][k] = 1.0;
                    }
                  else
                    {
                      casted_values[i][j][k] = 0.0;
                    }
                  break;
                case MESH_KIND_OUTPUT:
                  casted_values[i][j][k] = 0.0;
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
  };
}

// Function to free the memory allocated for the mesh
void
mesh_free (mesh_t *mesh)
{
  // Free the memory allocated for the 1D array
  free (mesh->values);
}

