#include "chrono.h"
#include "logging.h"
#include "stencil/comm_handler.h"
#include "stencil/config.h"
#include "stencil/init.h"
#include "stencil/mesh.h"
#include "stencil/solve.h"

#include <math.h>
#include <mpi.h>
#include <math.h>
#include <stdio.h>

static char *DEFAULT_CONFIG_PATH = "config.txt";
static char *DEFAULT_OUTPUT_PATH = NULL;

//
static void
save_results (FILE ofp[static 1], config_t const *cfg, mesh_t const *mesh,
              comm_handler_t const *comm_handler, duration_t elapsed)
{
  usz const mid_x = cfg->dim_x / 2;
  usz const mid_y = cfg->dim_y / 2;
  usz const mid_z = cfg->dim_z / 2;
  bool mid_x_is_in
      = (comm_handler->coord_x <= mid_x
         && mid_x < comm_handler->coord_x + comm_handler->loc_dim_x)
            ? true
            : false;
  bool mid_y_is_in
      = (comm_handler->coord_y <= mid_y
         && mid_y < comm_handler->coord_y + comm_handler->loc_dim_y)
            ? true
            : false;
  bool mid_z_is_in
      = (comm_handler->coord_z <= mid_z
         && mid_z < comm_handler->coord_z + comm_handler->loc_dim_z)
            ? true
            : false;

  f64 loc_elapsed_s = duration_as_s_f64 (elapsed);
  f64 loc_ns_per_elem = duration_as_ns_f64 (elapsed) / (f64)cfg->dim_x
                        / (f64)cfg->dim_y / (f64)cfg->dim_z;
  f64 glob_elapsed_s;
  f64 glob_ns_per_elem;

  i32 rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  i32 comm_size;
  MPI_Comm_size (MPI_COMM_WORLD, &comm_size);

  /* if (rank != 0) */
  MPI_Allreduce (&loc_elapsed_s, &glob_elapsed_s, 1, MPI_DOUBLE, MPI_SUM,
                 MPI_COMM_WORLD);
  MPI_Allreduce (&loc_ns_per_elem, &glob_ns_per_elem, 1, MPI_DOUBLE, MPI_SUM,
                 MPI_COMM_WORLD);

  if (mid_x_is_in && mid_y_is_in && mid_z_is_in)
    {
      // f64 ***mesh_values = (f64 ***)mesh->values;
      usz dim_y = mesh->dim_y;
      usz dim_z = mesh->dim_z;
      f64 (*mesh_values)[dim_y][dim_z] = (f64 (*)[dim_y][dim_z])mesh->values;

      fprintf (ofp, "%+18.15lf %12.9lf %12.3lf %zu %zu %zu\n",
               mesh_values[mid_x - comm_handler->coord_x + STENCIL_ORDER]
                          [mid_y - comm_handler->coord_y + STENCIL_ORDER]
                          [mid_z - comm_handler->coord_z + STENCIL_ORDER],

               glob_elapsed_s / (f64)comm_size,
               glob_ns_per_elem / (f64)comm_size, cfg->dim_x, cfg->dim_y,
               cfg->dim_z);
    }
}

//
i32
main (i32 argc, char *argv[argc + 1])
{
  MPI_Init (&argc, &argv);

  i32 rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  i32 comm_size;
  MPI_Comm_size (MPI_COMM_WORLD, &comm_size);

  char *config_path;
  char *output_path;

  usz BLOCK_SIZE_X = 4096;
  usz BLOCK_SIZE_Y = 128;
  usz BLOCK_SIZE_Z = 4096;


  if (2 == argc)
    {
      config_path = argv[1];
      output_path = DEFAULT_OUTPUT_PATH;
    }
  else if (3 == argc)
    {
      config_path = argv[1];
      output_path = argv[2];
    }
  else if (6 == argc)
    {
      config_path = argv[1];
      output_path = argv[2];
      BLOCK_SIZE_X = atoi (argv[3]);
      BLOCK_SIZE_Y = atoi (argv[4]);
      BLOCK_SIZE_Z = atoi (argv[5]);
    }

  else
    {
      config_path = DEFAULT_CONFIG_PATH;
      output_path = DEFAULT_OUTPUT_PATH;
    }

  config_t cfg = config_parse_from_file (config_path);

#ifndef NDEBUG
  if (rank == 0)
    {
      config_print (&cfg);
    }
#endif

  FILE *ofp;
  if (NULL != output_path)
    {
      ofp = fopen (output_path, "wb");
      if (NULL == ofp)
        {
          error ("failed to open output file `%s`", output_path);
        }
    }
  else
    {
      ofp = stdout;
    }

  comm_handler_t comm_handler = comm_handler_new (
      (u32)rank, (u32)comm_size, cfg.dim_x, cfg.dim_y, cfg.dim_z);

  // NOTE : Unrolled && Precomputed
  f64 pow_precomputed[STENCIL_ORDER] __attribute__ ((aligned (32)));
  for (usz o = 1; o <= STENCIL_ORDER; o += 4)
    {
      *(pow_precomputed + (o - 1)) = (1.0 / pow (17.0, (f64)(o)));
      *(pow_precomputed + (o)) = (1.0 / pow (17.0, (f64)(o + 1)));
      *(pow_precomputed + (o + 1)) = (1.0 / pow (17.0, (f64)(o + 2)));
      *(pow_precomputed + (o + 2)) = (1.0 / pow (17.0, (f64)(o + 3)));

    }
  else
    {
      ofp = stdout;
    }

  comm_handler_t comm_handler = comm_handler_new (
      (u32)rank, (u32)comm_size, cfg.dim_x, cfg.dim_y, cfg.dim_z);

#ifndef NDEBUG
  comm_handler_print (&comm_handler);
#endif
  usz dim_x = comm_handler.loc_dim_x;
  usz dim_y = comm_handler.loc_dim_y;
  usz dim_z = comm_handler.loc_dim_z;

  mesh_t A = mesh_new (dim_x, dim_y, dim_z, MESH_KIND_OUTPUT, &comm_handler);
  mesh_t B = mesh_new (dim_x, dim_y, dim_z, MESH_KIND_CONSTANT, &comm_handler);
  mesh_t C = mesh_new (dim_x, dim_y, dim_z, MESH_KIND_INPUT, &comm_handler);

  // Exchange ghost cells to make sure data is properly initialized
  // everywhere
  comm_handler_ghost_exchange (&comm_handler, &B);
  comm_handler_ghost_exchange (&comm_handler, &C);

  chrono_t chrono;

#ifndef NDEBUG
  if (rank == 0)
    {
      fprintf (stderr, "****************************************\n");
    }
#endif
#pragma omp parallel
  {
    for (usz it = 0; it < cfg.niter; ++it)
      {

#ifndef NDEBUG
        if (rank == 0)
          {
            fprintf (stderr, "Iteration #%2zu/%2zu\r", it + 1, cfg.niter);
          }
#endif

#pragma omp master
        {
          chrono_start (&chrono);
        }

        elementwise_multiply (&A, &B, &C);
        //   Compute Jacobi C=B@A (one iteration)
        solve_jacobi (&A, &C, pow_precomputed, BLOCK_SIZE_X, BLOCK_SIZE_Y,
                      BLOCK_SIZE_Z);

        // Exchange ghost cells for C meshes
        // No need to exchange A as its specific to a process
        // No need to exchange B as its a constant mesh

        comm_handler_ghost_exchange (&comm_handler, &C);

#pragma omp barrier

// Because ghost exchange is nowait and we had some sync problems
#pragma omp master
        {

          chrono_stop (&chrono);

          duration_t elapsed = chrono_elapsed (chrono);

          save_results (ofp, &cfg, &C, &comm_handler, elapsed);
        }
      }
  }

  mesh_drop (&A);
  mesh_drop (&B);
  mesh_drop (&C);
  fclose (ofp);

  MPI_Finalize ();
  return 0;
}

