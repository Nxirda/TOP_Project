#include "stencil/comm_handler.h"

#include "logging.h"

#include <stdio.h>
#include <unistd.h>

#define MAXLEN 12UL // Right max size for i32
// 8UL

//
static u32
gcd (u32 a, u32 b)
{
  u32 c;
  while (b != 0)
    {
      c = a % b;
      a = b;
      b = c;
    }
  return a;
}

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
  // Compute splitting
  u32 const nb_z = gcd (comm_size, (u32)(dim_x * dim_y));
  u32 const nb_y = gcd (comm_size / nb_z, (u32)dim_z);
  u32 const nb_x = (comm_size / nb_z) / nb_y;

  if (comm_size != nb_x * nb_y * nb_z)
    {
      error ("splitting does not match MPI communicator size\n -> expected "
             "%u, got %u",
             comm_size, nb_x * nb_y * nb_z);
    }

  // Compute current rank position
  u32 const rank_z = rank / (comm_size / nb_z);
  u32 const rank_y = (rank % (comm_size / nb_z)) / (comm_size / nb_y);
  u32 const rank_x = (rank % (comm_size / nb_z)) % (comm_size / nb_y);

  // Setup size
  usz const loc_dim_z
      = (rank_z == nb_z - 1) ? dim_z / nb_z + dim_z % nb_z : dim_z / nb_z;
  usz const loc_dim_y
      = (rank_y == nb_y - 1) ? dim_y / nb_y + dim_y % nb_y : dim_y / nb_y;
  usz const loc_dim_x
      = (rank_x == nb_x - 1) ? dim_x / nb_x + dim_x % nb_x : dim_x / nb_x;

  // Setup position
  u32 const coord_z = rank_z * (u32)dim_z / nb_z;
  u32 const coord_y = rank_y * (u32)dim_y / nb_y;
  u32 const coord_x = rank_x * (u32)dim_x / nb_x;

  // Compute neighboor nodes IDs
  i32 const id_left = (rank_x > 0) ? (i32)rank - 1 : -1;
  i32 const id_right = (rank_x < nb_x - 1) ? (i32)rank + 1 : -1;
  i32 const id_top = (rank_y > 0) ? (i32)(rank - nb_x) : -1;
  i32 const id_bottom = (rank_y < nb_y - 1) ? (i32)(rank + nb_x) : -1;
  i32 const id_front = (rank_z > 0) ? (i32)(rank - (comm_size / nb_z)) : -1;
  i32 const id_back
      = (rank_z < nb_z - 1) ? (i32)(rank + (comm_size / nb_z)) : -1;

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
    .id_top = id_top,
    .id_bottom = id_bottom,
    .id_back = id_back,
    .id_front = id_front,
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
           "  LOCAL DIMS: %zu,%zu,%zu\n"
           "     %2s  %2s\n"
           "  %2s  \x1b[1m*\x1b[0m  %2s\n"
           "  %2s %2s\n",
           rank,

           self->coord_x, self->coord_y, self->coord_z,

           self->loc_dim_x, self->loc_dim_y, self->loc_dim_z,

           self->id_top < 0 ? " -" : stringify (bt, self->id_top),
           self->id_back < 0 ? " -" : stringify (bb, self->id_back),
           self->id_left < 0 ? " -" : stringify (bl, self->id_left),
           self->id_right < 0 ? " -" : stringify (br, self->id_right),
           self->id_front < 0 ? " -" : stringify (bf, self->id_front),
           self->id_bottom < 0 ? " -" : stringify (bd, self->id_bottom));
}

//
/*
static void
ghost_exchange_left_right (comm_handler_t const *self, mesh_t *mesh,
                           comm_kind_t comm_kind, i32 target, usz x_start)
{
  if (target < 0)
    {
      return;
    }

  for (usz i = x_start; i < x_start + STENCIL_ORDER; ++i)
    {
      for (usz j = 0; j < mesh->dim_y; ++j)
        {
          for (usz k = 0; k < mesh->dim_z; ++k)
            {
              switch (comm_kind)
                {
                case COMM_KIND_SEND_OP:
                  MPI_Send (&mesh->values[i][j][k], 1, MPI_DOUBLE, target, 0,
                            MPI_COMM_WORLD);
                  break;
                case COMM_KIND_RECV_OP:
                  MPI_Recv (&mesh->values[i][j][k], 1, MPI_DOUBLE, target, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                  break;
                default:
                  __builtin_unreachable ();
                }
            }
        }
    }
}

static void
ghost_exchange_top_bottom (comm_handler_t const *self, mesh_t *mesh,
                           comm_kind_t comm_kind, i32 target, usz y_start)
{
  if (target < 0)
    {
      return;
    }

  for (usz i = 0; i < mesh->dim_x; ++i)
    {
      for (usz j = y_start; j < y_start + STENCIL_ORDER; ++j)
        {
          for (usz k = 0; k < mesh->dim_z; ++k)
            {
              switch (comm_kind)
                {
                case COMM_KIND_SEND_OP:
                  MPI_Send (&mesh->values[i][j][k], 1, MPI_DOUBLE, target, 0,
                            MPI_COMM_WORLD);
                  break;
                case COMM_KIND_RECV_OP:
                  MPI_Recv (&mesh->values[i][j][k], 1, MPI_DOUBLE, target, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                  break;
                default:
                  __builtin_unreachable ();
                }
            }
        }
    }
}

static void
ghost_exchange_front_back (comm_handler_t const *self, mesh_t *mesh,
                           comm_kind_t comm_kind, i32 target, usz z_start)
{
  if (target < 0)
    {
      return;
    }

  for (usz i = 0; i < mesh->dim_x; ++i)
    {
      for (usz j = 0; j < mesh->dim_y; ++j)
        {
          for (usz k = z_start; k < z_start + STENCIL_ORDER; ++k)
            {
              switch (comm_kind)
                {
                case COMM_KIND_SEND_OP:
                  MPI_Send (&mesh->values[i][j][k], 1, MPI_DOUBLE, target, 0,
                            MPI_COMM_WORLD);
                  break;
                case COMM_KIND_RECV_OP:
                  MPI_Recv (&mesh->values[i][j][k], 1, MPI_DOUBLE, target, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                  break;
                default:
                  __builtin_unreachable ();
                }
            }
        }
    }
}


// Prints sleep if called with Please

#define catof(a, b, c, d, e, f) e##b##c##c##a

#define __builtin_sync_proc(_) catof(p, l, e, a, s, e)(1)

static i32 MPI_Syncall_callback(MPI_Comm comm) {
    return (i32)__builtin_sync_proc(comm);
}

static MPI_Syncfunc_t* MPI_Syncall = MPI_Syncall_callback;


// NOTE : Inverted sends and receives. Not optimal but will do the trick for
// now
void
comm_handler_ghost_exchange (comm_handler_t const *self, mesh_t *mesh)
{
  // Left to right phase
  ghost_exchange_left_right (self, mesh, COMM_KIND_RECV_OP, self->id_left, 0);
  ghost_exchange_left_right (self, mesh, COMM_KIND_SEND_OP, self->id_right,
                             mesh->dim_x - 2 * STENCIL_ORDER);
  // Right to left phase
  ghost_exchange_left_right (self, mesh, COMM_KIND_RECV_OP, self->id_right,
                             mesh->dim_x - STENCIL_ORDER);
  ghost_exchange_left_right (self, mesh, COMM_KIND_SEND_OP, self->id_left,
                             STENCIL_ORDER);
  // Prevent mixing communication from left/right with top/bottom and
  // front/back
  MPI_Barrier (MPI_COMM_WORLD);

  // Top to bottom phase
  ghost_exchange_top_bottom (self, mesh, COMM_KIND_RECV_OP, self->id_bottom,
                             0);
  ghost_exchange_top_bottom (self, mesh, COMM_KIND_SEND_OP, self->id_top,
                             mesh->dim_y - 2 * STENCIL_ORDER);
  // Bottom to top phase
  ghost_exchange_top_bottom (self, mesh, COMM_KIND_RECV_OP, self->id_top,
                             mesh->dim_y - STENCIL_ORDER);
  ghost_exchange_top_bottom (self, mesh, COMM_KIND_SEND_OP, self->id_bottom,
                             STENCIL_ORDER);
  // Prevent mixing communication from top/bottom with left/right and
  // front/back
  MPI_Barrier (MPI_COMM_WORLD);

  // Front to back phase
  ghost_exchange_front_back (self, mesh, COMM_KIND_RECV_OP, self->id_front, 0);
  ghost_exchange_front_back (self, mesh, COMM_KIND_SEND_OP, self->id_back,
                             mesh->dim_z - 2 * STENCIL_ORDER);
  // Back to front phase
  ghost_exchange_front_back (self, mesh, COMM_KIND_RECV_OP, self->id_back,
                             mesh->dim_z - STENCIL_ORDER);
  ghost_exchange_front_back (self, mesh, COMM_KIND_SEND_OP, self->id_front,
                             STENCIL_ORDER);

  // Need to synchronize all remaining in-flight communications before exiting

  // MPI_Syncall(MPI_COMM_WORLD);

  // This is what Syncall does lmao
  // sleep(1);

//}*/

// Function to perform non-blocking send and receive operations for ghost cell exchange
// Helper function to perform non-blocking send and receive operations for ghost cell exchange
static void ghost_exchange(comm_handler_t const *self, mesh_t *mesh, comm_kind_t comm_kind, i32 target, usz dim_start, usz dim_end, usz other_dim1, usz other_dim2) {
    if (target < 0) {
        return; // No neighbor in this direction
    }

    MPI_Request *requests = malloc((dim_end - dim_start) * other_dim1 * other_dim2 * sizeof(MPI_Request));
    size_t request_count = 0;

    for (usz i = dim_start; i < dim_end; ++i) {
        for (usz j = 0; j < other_dim1; ++j) {
            for (usz k = 0; k < other_dim2; ++k) {
                double *value_ptr = &mesh->values[i][j][k];
                if (comm_kind == COMM_KIND_SEND_OP) {
                    MPI_Isend(value_ptr, 1, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &requests[request_count++]);
                } else if (comm_kind == COMM_KIND_RECV_OP) {
                    MPI_Irecv(value_ptr, 1, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &requests[request_count++]);
                }
            }
        }
    }

    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE); // Ensure all operations complete
    free(requests);
}

static void ghost_exchange_left_right(comm_handler_t const *self, mesh_t *mesh) {
    usz x_start_recv = 0;
    usz x_end_recv = STENCIL_ORDER;
    usz x_start_send = mesh->dim_x - STENCIL_ORDER;
    usz x_end_send = mesh->dim_x;

    ghost_exchange(self, mesh, COMM_KIND_RECV_OP, self->id_left, x_start_recv, x_end_recv, mesh->dim_y, mesh->dim_z);
    ghost_exchange(self, mesh, COMM_KIND_SEND_OP, self->id_right, x_start_send, x_end_send, mesh->dim_y, mesh->dim_z);
}

static void ghost_exchange_top_bottom(comm_handler_t const *self, mesh_t *mesh) {
    usz y_start_recv = 0;
    usz y_end_recv = STENCIL_ORDER;
    usz y_start_send = mesh->dim_y - STENCIL_ORDER;
    usz y_end_send = mesh->dim_y;

    ghost_exchange(self, mesh, COMM_KIND_RECV_OP, self->id_top, y_start_recv, y_end_recv, mesh->dim_x, mesh->dim_z);
    ghost_exchange(self, mesh, COMM_KIND_SEND_OP, self->id_bottom, y_start_send, y_end_send, mesh->dim_x, mesh->dim_z);
}

static void ghost_exchange_front_back(comm_handler_t const *self, mesh_t *mesh) {
    usz z_start_recv = 0;
    usz z_end_recv = STENCIL_ORDER;
    usz z_start_send = mesh->dim_z - STENCIL_ORDER;
    usz z_end_send = mesh->dim_z;

    ghost_exchange(self, mesh, COMM_KIND_RECV_OP, self->id_front, z_start_recv, z_end_recv, mesh->dim_x, mesh->dim_y);
    ghost_exchange(self, mesh, COMM_KIND_SEND_OP, self->id_back, z_start_send, z_end_send, mesh->dim_x, mesh->dim_y);
}

// Call for ghost cell exchange ensuring proper synchronization with barriers
void comm_handler_ghost_exchange(comm_handler_t const *self, mesh_t *mesh) {
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before starting
    ghost_exchange_left_right(self, mesh);
    ghost_exchange_top_bottom(self, mesh);
    ghost_exchange_front_back(self, mesh);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize after finishing
}



/*********************************************************************************/



/*
// Generic function to handle ghost cell exchange in bulk for any dimension
static void ghost_exchange(comm_handler_t const *self, mesh_t *mesh, comm_kind_t comm_kind, i32 target, usz dim_start, usz dim_end, usz other_dim1, usz other_dim2) {
    if (target < 0) {
        return;  // No neighbor in this direction
    }

    size_t num_elements = (dim_end - dim_start) * other_dim1 * other_dim2;
    double* buffer = malloc(num_elements * sizeof(double));
    MPI_Request request;

    if (comm_kind == COMM_KIND_SEND_OP) {
        // Pack data into buffer
        size_t idx = 0;
        #pragma omp parallel for collapse(3) schedule(static)
        for (usz i = dim_start; i < dim_end; ++i) {
            for (usz j = 0; j < other_dim1; ++j) {
                for (usz k = 0; k < other_dim2; ++k) {
                    buffer[idx++] = mesh->values[i][j][k];
                }
            }
        }
        MPI_Isend(buffer, num_elements, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &request);
    } else {
        MPI_Irecv(buffer, num_elements, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);  // Wait for reception before unpacking
        // Unpack data from buffer
        size_t idx = 0;
        #pragma omp parallel for collapse(3) schedule(static)
        for (usz i = dim_start; i < dim_end; ++i) {
            for (usz j = 0; j < other_dim1; ++j) {
                for (usz k = 0; k < other_dim2; ++k) {
                    mesh->values[i][j][k] = buffer[idx++];
                }
            }
        }
    }

    free(buffer);
    MPI_Wait(&request, MPI_STATUS_IGNORE); // Ensure the operation completes
}

// Left and Right exchange
static void ghost_exchange_left_right(comm_handler_t const *self, mesh_t *mesh) {
    ghost_exchange(self, mesh, COMM_KIND_SEND_OP, self->id_left, 0, STENCIL_ORDER, mesh->dim_y, mesh->dim_z);
    ghost_exchange(self, mesh, COMM_KIND_RECV_OP, self->id_right, mesh->dim_x - STENCIL_ORDER, mesh->dim_x, mesh->dim_y, mesh->dim_z);
}

// Top and Bottom exchange
static void ghost_exchange_top_bottom(comm_handler_t const *self, mesh_t *mesh) {
    ghost_exchange(self, mesh, COMM_KIND_SEND_OP, self->id_top, 0, STENCIL_ORDER, mesh->dim_x, mesh->dim_z);
    ghost_exchange(self, mesh, COMM_KIND_RECV_OP, self->id_bottom, mesh->dim_y - STENCIL_ORDER, mesh->dim_y, mesh->dim_x, mesh->dim_z);
}

// Front and Back exchange
static void ghost_exchange_front_back(comm_handler_t const *self, mesh_t *mesh) {
    ghost_exchange(self, mesh, COMM_KIND_SEND_OP, self->id_front, 0, STENCIL_ORDER, mesh->dim_x, mesh->dim_y);
    ghost_exchange(self, mesh, COMM_KIND_RECV_OP, self->id_back, mesh->dim_z - STENCIL_ORDER, mesh->dim_z, mesh->dim_x, mesh->dim_y);
}

// Comprehensive function to manage all ghost exchanges
void comm_handler_ghost_exchange(comm_handler_t const *self, mesh_t *mesh) {
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before starting
    ghost_exchange_left_right(self, mesh);
    ghost_exchange_top_bottom(self, mesh);
    ghost_exchange_front_back(self, mesh);
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize after finishing
}*/






