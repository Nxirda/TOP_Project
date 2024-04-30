#include "stencil/comm_handler.h"

#include "logging.h"

#include <omp.h>
#include <stdio.h>
#include <unistd.h>

#define MAXLEN 12UL // Right max size for i32

//
static char *
stringify (char buf[static MAXLEN], i32 num)
{
  snprintf (buf, MAXLEN, "%d", num);
  return buf;
}

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
  static char bt[MAXLEN];
  static char bb[MAXLEN];
  static char bl[MAXLEN];
  static char br[MAXLEN];
  static char bf[MAXLEN];
  static char bd[MAXLEN];
  fprintf (stderr,
           "****************************************\n"
           "RANK %d:\n"
           "  COORDS:     %u,%u,%u\n"
           "  LOCAL DIMS: %zu,%zu,%zu\n",
           rank, self->coord_x, self->coord_y, self->coord_z, self->loc_dim_x,
           self->loc_dim_y, self->loc_dim_z,
           self->id_left < 0 ? " -" : stringify (bl, self->id_left),
           self->id_right < 0 ? " -" : stringify (br, self->id_right));
}

//
static void
send_receive (comm_handler_t const *self, mesh_t *mesh, comm_kind_t comm_kind,
              i32 target, usz x_start, MPI_Request *requests,
              usz *request_index)
{
  if (target < 0)
    {
      return;
    }
  usz req = 0;
  for (usz i = x_start; i < x_start + STENCIL_ORDER; ++i)
    {
      for (usz j = 0; j < mesh->dim_y; ++j)
        {

          req++;

          switch (comm_kind)
            {
            case COMM_KIND_SEND_OP:
              MPI_Isend (&mesh->values[i][j][0], mesh->dim_z, MPI_DOUBLE,
                         target, 0, MPI_COMM_WORLD,
                         &requests[(*request_index)]);
              break;
            case COMM_KIND_RECV_OP:
              MPI_Irecv (&mesh->values[i][j][0], mesh->dim_z, MPI_DOUBLE,
                         target, 0, MPI_COMM_WORLD,
                         &requests[(*request_index)]);
              break;
            default:
              __builtin_unreachable ();
            }
#pragma omp atomic
          (*request_index)++;
        }
    }
}

//
void
comm_handler_ghost_exchange (comm_handler_t const *self, mesh_t *mesh)
{

#pragma omp single nowait
  {
    // 4 because we do 4 send/recv
    MPI_Request requests[4 * STENCIL_ORDER * mesh->dim_y];
    MPI_Status status[4 * STENCIL_ORDER * mesh->dim_y];
    usz req_idx = 0;

    send_receive (self, mesh, COMM_KIND_RECV_OP, self->id_left, 0, requests,
                  &req_idx);

    send_receive (self, mesh, COMM_KIND_SEND_OP, self->id_left, STENCIL_ORDER,
                  requests, &req_idx);

    send_receive (self, mesh, COMM_KIND_SEND_OP, self->id_right,
                  mesh->dim_x - (2 * STENCIL_ORDER), requests, &req_idx);

    send_receive (self, mesh, COMM_KIND_RECV_OP, self->id_right,
                  mesh->dim_x - STENCIL_ORDER, requests, &req_idx);

    MPI_Waitall (req_idx, requests, status);
  }
}