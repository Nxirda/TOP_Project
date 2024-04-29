#include "stencil/mesh.h"

#include "logging.h"

#include <assert.h>
#include <stdlib.h>

//
void
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
}