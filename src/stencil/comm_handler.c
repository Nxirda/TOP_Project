#include "stencil/comm_handler.h"

#include "logging.h"

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <unistd.h>

#define MAXLEN 12UL // Right max size for i32

//
comm_handler_t
comm_handler_new (u32 rank, u32 comm_size, usz dim_x, usz dim_y, usz dim_z)
{
  // Number of proc per axis
  u32 const nb_x = comm_size;
  u32 const nb_y = 1;
  u32 const nb_z = 1;

  if (comm_size != nb_x * nb_y * nb_z)
    {
      error ("splitting does not match MPI communicator size\n -> expected "
             "%u, got %u",
             comm_size, nb_x * nb_y * nb_z);
    }

  // Setup size (only splitted on x axis)
  usz const loc_dim_z = dim_z;
  usz const loc_dim_y = dim_y;
  usz const loc_dim_x
      = (rank == nb_x - 1) ? dim_x / nb_x + dim_x % nb_x : dim_x / nb_x;

  // Setup position
  u32 const coord_z = 0;
  u32 const coord_y = 0;
  u32 const coord_x = rank * ((u32)dim_x / nb_x);

  // Compute neighboor nodes IDs
  i32 const id_left = (rank > 0) ? (i32)rank - 1 : -1;
  i32 const id_right = (rank < nb_x - 1) ? (i32)rank + 1 : -1;

  return (comm_handler_t){
    .nb_x = nb_x,
    .nb_y = nb_y,
    .nb_z = nb_z,
    .coord_x = coord_x,
    .coord_y = coord_y,
    .coord_z = coord_z,
    .loc_dim_x = loc_dim_x,
    .loc_dim_y = loc_dim_y,
    .loc_dim_z = loc_dim_z,
    .id_left = id_left,
    .id_right = id_right,
  };
}

//
void
comm_handler_print (comm_handler_t const *self)
{
  i32 rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  fprintf (stderr,
           "****************************************\n"
           "RANK %d:\n"
           "  COORDS:     %u,%u,%u\n"
           "  LOCAL DIMS: %zu,%zu,%zu\n",
           rank, self->coord_x, self->coord_y, self->coord_z, self->loc_dim_x,
           self->loc_dim_y, self->loc_dim_z);
}

//
static void
pack_buffer (mesh_t *mesh, usz x_start, f64 *send_buffer)
{
  f64 (*mesh_values)[mesh->dim_y][mesh->dim_z]
      = make_3dspan (f64, const, mesh->values, mesh->dim_y, mesh->dim_z);
  usz idx = 0;
  for (usz i = x_start; i < x_start + STENCIL_ORDER; i++)
    {
      for (usz j = 0; j < mesh->dim_y; j++)
        {
          for (usz k = 0; k < mesh->dim_z; k++)
            {
              send_buffer[idx++] = mesh_values[i][j][k];
            }
        }
    }
}

//
static void
unpack_buffer (mesh_t *mesh, usz x_start, f64 *recv_buffer)
{
  f64 (*mesh_values)[mesh->dim_y][mesh->dim_z]
      = make_3dspan (f64, , mesh->values, mesh->dim_y, mesh->dim_z);
  usz idx = 0;
  for (usz i = x_start; i < x_start + STENCIL_ORDER; i++)
    {
      for (usz j = 0; j < mesh->dim_y; j++)
        {
          for (usz k = 0; k < mesh->dim_z; k++)
            {
              mesh_values[i][j][k] = recv_buffer[idx++];
            }
        }
    }
}

//
void
comm_handler_ghost_exchange (comm_handler_t const *self, mesh_t *mesh)
{

    i32 size;
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    if(size == 1)
      return;

#pragma omp single
  {
    usz data_size = (2 * STENCIL_ORDER) * mesh->dim_y * mesh->dim_z;

    f64 *left_buffer = (f64 *)malloc (data_size * sizeof (f64));
    f64 *right_buffer = (f64 *)malloc (data_size * sizeof (f64));
    ;

    f64 *recv_left_buffer = (f64 *)malloc (data_size * sizeof (f64));
    f64 *recv_right_buffer = (f64 *)malloc (data_size * sizeof (f64));

    if (self->id_left >= 0)
      {
        pack_buffer (mesh, STENCIL_ORDER, left_buffer);
      }

    // Because -1 by default
    if (self->id_right < size && self->id_right > 0)
      {
        pack_buffer (mesh, mesh->dim_x - (2 * STENCIL_ORDER), right_buffer);
      }

    MPI_Sendrecv (left_buffer, data_size, MPI_DOUBLE, self->id_left, 0,
                  recv_right_buffer, data_size, MPI_DOUBLE, self->id_right, 0,
                  MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv (right_buffer, data_size, MPI_DOUBLE, self->id_right, 0,
                  recv_left_buffer, data_size, MPI_DOUBLE, self->id_left, 0,
                  MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (self->id_right < size && self->id_right > 0)
      {
        unpack_buffer (mesh, mesh->dim_x - STENCIL_ORDER, recv_right_buffer);
      }

    if (self->id_left >= 0)
      {
        unpack_buffer (mesh, 0, recv_left_buffer);
      }

    free (left_buffer);
    free (right_buffer);
    free (recv_left_buffer);
    free (recv_right_buffer);
  }
}