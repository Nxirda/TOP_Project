#include "stencil/mesh.h"

#include "logging.h"

#include <assert.h>
#include <stdlib.h>

void
mesh_drop (mesh_t *self)
{
  if (self->values != NULL)
    free (self->values);
}