/* typedef struct comm_handler_s
{
  usz loc_dim_x;
  usz loc_dim_y;
  usz loc_dim_z;

  u32 coord_x;
  u32 coord_y;
  u32 coord_z;

  MPI_Comm comm;
  
} __attribute__ ((packed)) comm_handler_t; */

/* comm_handler_t comm_handler_new (u32 rank, u32 comm_size, usz dim_x, usz dim_y, usz dim_z);
 */
/* void ghost_exchange(comm_handler_t const *self, mesh_t *mesh, f64 *send_buffer, i32 pack_size);
 */

 //
void
solve_jacobi_2 (usz dim_x, usz dim_y, usz dim_z, f64 (*A_Matrix)[dim_y][dim_z],
                f64 (*B_Matrix)[dim_y][dim_z], f64 (*C_Matrix)[dim_y][dim_z],
                f64 pow_precomputed[STENCIL_ORDER])
{
  /* assert (A->dim_x == B->dim_x && B->dim_x == C->dim_x);
  assert (A->dim_y == B->dim_y && B->dim_y == C->dim_y);
  assert (A->dim_z == B->dim_z && B->dim_z == C->dim_z); */

  usz const lim_x = dim_x - STENCIL_ORDER;
  usz const lim_y = dim_y - STENCIL_ORDER;
  usz const lim_z = dim_z - STENCIL_ORDER;

  /*  usz dim_x = dim_x;
   usz dim_y = dim_y;
   usz dim_z = dim_z; */
  __m256d pow_low = _mm256_load_pd (pow_precomputed);
  __m256d pow_high = _mm256_load_pd (4 + pow_precomputed);

  /* f64 (*A_Matrix)[dim_y][dim_z] = (f64 (*)[dim_y][dim_z])A->values;
  f64 (*B_Matrix)[dim_y][dim_z] = (f64 (*)[dim_y][dim_z])B->values;
  f64 (*C_Matrix)[dim_y][dim_z] = (f64 (*)[dim_y][dim_z])C->values; */

#pragma omp for schedule(static, 8)
  for (usz i = STENCIL_ORDER; i < lim_x; ++i)
    {
      for (usz j = STENCIL_ORDER; j < lim_y; ++j)
        {
          for (usz k = STENCIL_ORDER; k < lim_z; ++k)
            {
              __m256d result;

              // Negative A neighbors on x axis
              __m256d niA5_8 = _mm256_set_pd (
                  A_Matrix[i - 8][j][k], A_Matrix[i - 7][j][k],
                  A_Matrix[i - 6][j][k], A_Matrix[i - 5][j][k]);

              __m256d niA1_4 = _mm256_set_pd (
                  A_Matrix[i - 4][j][k], A_Matrix[i - 3][j][k],
                  A_Matrix[i - 2][j][k], A_Matrix[i - 1][j][k]);

              result = _mm256_mul_pd (niA5_8, pow_high);
              result = _mm256_fmadd_pd (niA1_4, pow_low, result);

              // Positive A neighbors on x axis
              __m256d piA1_4 = _mm256_set_pd (
                  A_Matrix[i + 4][j][k], A_Matrix[i + 3][j][k],
                  A_Matrix[i + 2][j][k], A_Matrix[i + 1][j][k]);

              __m256d piA5_8 = _mm256_set_pd (
                  A_Matrix[i + 8][j][k], A_Matrix[i + 7][j][k],
                  A_Matrix[i + 6][j][k], A_Matrix[i + 5][j][k]);

              result = _mm256_fmadd_pd (piA1_4, pow_low, result);
              result = _mm256_fmadd_pd (piA5_8, pow_high, result);

              // Negative A neighbors on y axis
              __m256d njA5_8 = _mm256_set_pd (
                  A_Matrix[i][j - 8][k], A_Matrix[i][j - 7][k],
                  A_Matrix[i][j - 6][k], A_Matrix[i][j - 5][k]);

              __m256d njA1_4 = _mm256_set_pd (
                  A_Matrix[i][j - 4][k], A_Matrix[i][j - 3][k],
                  A_Matrix[i][j - 2][k], A_Matrix[i][j - 1][k]);

              result = _mm256_fmadd_pd (njA5_8, pow_high, result);
              result = _mm256_fmadd_pd (njA1_4, pow_low, result);

              // Positive A neighbors on y axis
              __m256d pjA1_4 = _mm256_set_pd (
                  A_Matrix[i][j + 4][k], A_Matrix[i][j + 3][k],
                  A_Matrix[i][j + 2][k], A_Matrix[i][j + 1][k]);

              __m256d pjA5_8 = _mm256_set_pd (
                  A_Matrix[i][j + 8][k], A_Matrix[i][j + 7][k],
                  A_Matrix[i][j + 6][k], A_Matrix[i][j + 5][k]);

              result = _mm256_fmadd_pd (pjA1_4, pow_low, result);
              result = _mm256_fmadd_pd (pjA5_8, pow_high, result);

              // Negative A neighbors on z axis
              __m256d nkA5_8 = _mm256_set_pd (
                  A_Matrix[i][j][k - 8], A_Matrix[i][j][k - 7],
                  A_Matrix[i][j][k - 6], A_Matrix[i][j][k - 5]);

              __m256d nkA1_4 = _mm256_set_pd (
                  A_Matrix[i][j][k - 4], A_Matrix[i][j][k - 3],
                  A_Matrix[i][j][k - 2], A_Matrix[i][j][k - 1]);

              result = _mm256_fmadd_pd (nkA5_8, pow_high, result);
              result = _mm256_fmadd_pd (nkA1_4, pow_low, result);

              // Positive A neighbors on z axis
              __m256d pkA1_4 = _mm256_set_pd (
                  A_Matrix[i][j][k + 4], A_Matrix[i][j][k + 3],
                  A_Matrix[i][j][k + 2], A_Matrix[i][j][k + 1]);

              __m256d pkA5_8 = _mm256_set_pd (
                  A_Matrix[i][j][k + 8], A_Matrix[i][j][k + 7],
                  A_Matrix[i][j][k + 6], A_Matrix[i][j][k + 5]);

              result = _mm256_fmadd_pd (pkA1_4, pow_low, result);
              result = _mm256_fmadd_pd (pkA5_8, pow_high, result);

              __m128d low = _mm256_extractf128_pd (result, 0);
              __m128d high = _mm256_extractf128_pd (result, 1);

              __m128d sum = _mm_add_pd (high, low);
              sum = _mm_hadd_pd (sum, sum);

              C_Matrix[i][j][k] = A_Matrix[i][j][k] + _mm_cvtsd_f64 (sum);
            }
        }
    }
}

//
void
elementwise_multiply_2 (usz dim_x, usz dim_y, usz dim_z,
                        f64 (*A_Matrix)[dim_y][dim_z],
                        f64 (*B_Matrix)[dim_y][dim_z],
                        f64 (*C_Matrix)[dim_y][dim_z])
{
  /* usz j, k;
  usz dim_x = C->dim_x;
  usz dim_y = C->dim_y;
  usz dim_z = C->dim_z;
  f64 (*A_Matrix)[dim_y][dim_z] = (f64 (*)[dim_y][dim_z])A->values;
  f64 (*B_Matrix)[dim_y][dim_z] = (f64 (*)[dim_y][dim_z])B->values;
  f64 (*C_Matrix)[dim_y][dim_z] = (f64 (*)[dim_y][dim_z])C->values; */
  usz j, k;

#pragma omp for schedule(static, 8)
  for (usz i = 0; i < dim_x; i++)
    {
      for (j = 0; j < dim_y; j++)
        {
          for (k = 0; k < dim_z; k++)
            {
              A_Matrix[i][j][k] = C_Matrix[i][j][k] * B_Matrix[i][j][k];
            }
        }
    }
}

//
/* void
mesh_drop (mesh_t *self)
{
  if (NULL != self->values)
    {
      for (usz i = 0; i < self->dim_x; ++i)
        {
          for (usz j = 0; j < self->dim_y; ++j)
            {
              free (self->values[i][j]);
            }
          free (self->values[i]);
        }
      free (self->values);
    }
} */

/* mesh_t
mesh_new_2 (usz base_dim_x, usz base_dim_y, usz base_dim_z, mesh_kind_t kind,
            comm_handler_t const *comm_handler)
{
  usz const ghost_size = 2 * STENCIL_ORDER;
  usz dim_x = (base_dim_x + ghost_size);
  usz dim_y = (base_dim_y + ghost_size);
  usz dim_z = (base_dim_z + ghost_size);

  f64 ***values = (f64 ***)malloc (dim_x * sizeof (f64 **));

  // A->values = (f64 ***)malloc (A->dim_x * sizeof (f64 **));

  f64 *values = (f64*)malloc(dim_x * dim_y * dim_z * sizeof(f64));

  if (NULL == values)
    {
      error ("failed to allocate mesh of size %zu bytes",
             dim_x * dim_y * dim_z);
    }
#pragma omp parallel for schedule(static, 8)
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
          values[i][j] = (f64 *)malloc (dim_z * sizeof (f64));

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
                      && (k >= STENCIL_ORDER && (k < dim_z - STENCIL_ORDER)))
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
  f64 *casted_values = (f64 *)values;
  // casted_values = (f64 *)values;
  //free (values);
  return (mesh_t){
    .dim_x = dim_x,
     .dim_y = dim_y,
     .dim_z = dim_z,
     .values = casted_values,
  };
} */

//
/*comm_handler_t
comm_handler_new (u32 rank, u32 comm_size, usz dim_x, usz dim_y, usz dim_z)
{
  i32 ndims = DIMS;
  i32 dims[3] = {0,0,0};
  i32 periods[DIMS] = {0,0,0};
  //MPI_Dims_create(comm_size, ndims, dims);

  while (MPI_Dims_create(comm_size, ndims, dims) != MPI_SUCCESS) {
    // Decrease the number of processes
    comm_size--;

    // Check if the number of processes is too small
    if (comm_size < ndims) {
        // Handle the case where a valid decomposition cannot be found
        fprintf(stderr, "Error: Failed to find a valid decomposition for %d processes and %d dimensions.\n", comm_size, ndims);
        exit(EXIT_FAILURE);
    }
}

  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &cart_comm);

  i32 cart_rank;
  MPI_Comm_rank(cart_comm, &cart_rank);

  usz const loc_dim_x = dim_x / dims[0];
  usz const loc_dim_y = dim_y / dims[1];
  usz const loc_dim_z = dim_z / dims[2];

  int coords[DIMS]; // coordinates of the process in the Cartesian grid
  MPI_Cart_coords(cart_comm, cart_rank, 3, coords);

  usz const start_x = coords[0] * loc_dim_x; 
  usz const start_y = coords[1] * loc_dim_y; 
  usz const start_z = coords[2] * loc_dim_z;
  
  return (comm_handler_t){
      .loc_dim_x = loc_dim_x,
      .loc_dim_y = loc_dim_y,
      .loc_dim_z = loc_dim_z,
      .coord_x = start_x,
      .coord_y = start_y,
      .coord_z = start_z,
      .comm = cart_comm,
    };
}
 */

// Pack_size is per process
/* void ghost_exchange(comm_handler_t const *self, mesh_t *mesh, f64 *send_buffer, i32 pack_size){
  // Calculate the total number of cells in the mesh
  //usz total_cells = mesh->dim_x * mesh->dim_y * mesh->dim_z;

  i32 size,rank, coords[DIMS], dims[DIMS], periods[DIMS];
  
  MPI_Comm_size(self->comm, &size);
  MPI_Comm_rank(self->comm, &rank);
  MPI_Cart_get(self->comm, DIMS, dims, periods, coords);
  
  i32* send_counts = (i32*)malloc(size * sizeof(i32));
  i32* send_displs = (i32*)malloc(size * sizeof(i32));

  //f64* recv_buffer = (f64*)malloc(pack_size * size * sizeof(f64));

  // Calculate send counts and displacements
  for (i32 i = 0; i < size; ++i) {
      send_counts[i] = pack_size;
      send_displs[i] = i * pack_size;
  }

  i32 neighbor_rank_left, neighbor_rank_right;
  i32 neighbor_rank_down, neighbor_rank_up;
  i32 neighbor_rank_front, neighbor_rank_back;

  MPI_Cart_shift(self->comm, 0, 1, &neighbor_rank_left, &neighbor_rank_right);
  MPI_Cart_shift(self->comm, 1, 1, &neighbor_rank_up, &neighbor_rank_down);
  MPI_Cart_shift(self->comm, 2, 1, &neighbor_rank_front, &neighbor_rank_back);

  if (neighbor_rank_left >= 0) {
      send_counts[neighbor_rank_left] = pack_size;
      send_displs[neighbor_rank_left] = 0;
  }
  if (neighbor_rank_right >= 0) {
      send_counts[neighbor_rank_right] = pack_size;
      send_displs[neighbor_rank_right] = pack_size;
  }
  if (neighbor_rank_up >= 0) {
      send_counts[neighbor_rank_up] = pack_size;
      send_displs[neighbor_rank_up] = 2 * pack_size;
  }
  if (neighbor_rank_down >= 0) {
      send_counts[neighbor_rank_down] = pack_size;
      send_displs[neighbor_rank_down] = 3 * pack_size;
  }
  if (neighbor_rank_front >= 0) {
      send_counts[neighbor_rank_front] = pack_size;
      send_displs[neighbor_rank_front] = 4 * pack_size;
  }
  if (neighbor_rank_back >= 0) {
      send_counts[neighbor_rank_back] = pack_size;
      send_displs[neighbor_rank_back] = 5 * pack_size;
  }

  MPI_Alltoallv(send_buffer, send_counts, send_displs, MPI_DOUBLE, 
                send_buffer, send_counts, send_displs, MPI_DOUBLE, self->comm);  

} */

// Function to free the memory allocated for the mesh
/* void
mesh_drop (mesh_t *self)
{
  if (NULL != self->values)
    {
      // Cast back to f64*** to free the memory properly
      f64 ***values = (f64 ***)self->values;
      for (usz i = 0; i < self->dim_x; ++i)
        {
          for (usz j = 0; j < self->dim_y; ++j)
            {
              free (values[i][j]);
            }
          free (values[i]);
        }
      free (values);
    }
} */

//
static void
send_receive (comm_handler_t const *self, mesh_t *mesh, comm_kind_t comm_kind,
              i32 target, usz x_start, MPI_Request *requests,
              usz *request_index)
{
  f64 (*mesh_values)[mesh->dim_y][mesh->dim_z] = make_3dspan (f64, , mesh->values, mesh->dim_y, mesh->dim_z);

  if (target < 0)
    {
      return;
    }
  //usz req = 0;

  usz data_size = STENCIL_ORDER * mesh->dim_y * mesh->dim_z;
 /*  usz idx = 0;
  if(comm_kind = )
  for (usz i = x_start; i < x_start + STENCIL_ORDER; ++i)
    {
      for (usz j = 0; j < mesh->dim_y; ++j)
        {
          for (usz k = 0; k < mesh->dim_z; ++k)
          {
            send_buffer[idx++] = mesh_values[i][j][k];
          } */
         /*  req++;

          switch (comm_kind)
            {
            case COMM_KIND_SEND_OP:
              MPI_Isend (&mesh_values [i][j][0], mesh->dim_z, MPI_DOUBLE,
                         target, 0, MPI_COMM_WORLD,
                         &requests[(*request_index)]);
              break;
            case COMM_KIND_RECV_OP:
              MPI_Irecv (&mesh_values [i][j][0], mesh->dim_z, MPI_DOUBLE,
                         target, 0, MPI_COMM_WORLD,
                         &requests[(*request_index)]);
              break;
            default:
              __builtin_unreachable ();
            }
#pragma omp atomic
          (*request_index)++; */
    /*     }
    } */

     // Perform MPI communication
   /*  switch (comm_kind) {
      case COMM_KIND_SEND_OP:
          MPI_Isend(send_buffer, data_size, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &requests[(*request_index)]);
          break;
      case COMM_KIND_RECV_OP:
          MPI_Irecv(send_buffer, data_size, MPI_DOUBLE, target, 0, MPI_COMM_WORLD, &requests[(*request_index)]);
          break;
      default:
          __builtin_unreachable();
    } */

/* #pragma omp atomic
    (*request_index)++;

  if (comm_kind == COMM_KIND_RECV_OP) {
      // Unpack data from the received buffer and update the mesh
      idx = 0;
      for (usz i = x_start; i < x_start + STENCIL_ORDER; ++i) {
          for (usz j = 0; j < mesh->dim_y; ++j) {
              for (usz k = 0; k < mesh->dim_z; ++k) {
                  mesh_values[i][j][k] = send_buffer[idx++];
              }
          }
      }
  } */
  //free(send_buffer);
} 

//
/* static void setup_mesh_cell_kinds(mesh_t* mesh) {
    for (usz i = 0; i < mesh->dim_x; ++i) {
        for (usz j = 0; j < mesh->dim_y; ++j) {
            for (usz k = 0; k < mesh->dim_z; ++k) {
                mesh->cells_kind[i][j][k] = mesh_set_cell_kind(mesh, i, j,
k);
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

////////////////////////////////////
void elementwise_multiply_2 (usz dim_x, usz dim_y, usz dim_z,
                             f64 (*A_Matrix)[dim_y][dim_z],
                             f64 (*B_Matrix)[dim_y][dim_z],
                             f64 (*C_Matrix)[dim_y][dim_z]);

void solve_jacobi_2 (usz dim_x, usz dim_y, usz dim_z,
                     f64 (*A_Matrix)[dim_y][dim_z],
                     f64 (*B_Matrix)[dim_y][dim_z],
                     f64 (*C_Matrix)[dim_y][dim_z],
                     f64 pow_precomputed[STENCIL_ORDER]);
