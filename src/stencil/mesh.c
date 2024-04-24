#include "stencil/mesh.h"
#include "logging.h"

#include <assert.h>
#include <stdlib.h>


mesh_t mesh_new(usz dim_x, usz dim_y, usz dim_z, mesh_kind_t kind) {
    usz ghost_size = 2 * STENCIL_ORDER;
    usz total_cells = (dim_x + ghost_size) * (dim_y + ghost_size) * (dim_z + ghost_size);

    f64* values = malloc(total_cells * sizeof(f64));
    if (!values) {
        error("Failed to allocate memory for mesh values.");
    }

    cell_kind_t* kinds = malloc(total_cells * sizeof(cell_kind_t));
    if (!kinds) {
        free(values);  // Ensure we don't leak memory if the second allocation fails.
        error("Failed to allocate memory for mesh cell kinds.");
    }

    // Initialize values and kinds arrays
    for (usz i = 0; i < total_cells; ++i) {
        values[i] = 0.0; // Default initialization to 0.0 for values
        kinds[i] = CELL_KIND_PHANTOM; // Default to phantom, will be set correctly later
    }

    return (mesh_t){
        .dim_x = dim_x + ghost_size,
        .dim_y = dim_y + ghost_size,
        .dim_z = dim_z + ghost_size,
        .values = values,
        .kinds = kinds,
        .kind = kind
    };
}

void mesh_drop(mesh_t* self) {
    if (self) {
        free(self->values);
        free(self->kinds);
    }
}

static char const* mesh_kind_as_str(mesh_t const* self) {
    static char const* MESH_KINDS_STR[] = {
        "CONSTANT",
        "INPUT",
        "OUTPUT",
    };
    return MESH_KINDS_STR[(usz)self->kind];
}

void mesh_print(const mesh_t* self, const char* name) {
    fprintf(
        stderr,
        "****************************************\n"
        "MESH `%s`\n\tKIND: %s\n\tDIMS: %zux%zux%zu\n\tVALUES:\n",
        name,
        mesh_kind_as_str(self),
        self->dim_x,
        self->dim_y,
        self->dim_z
    );

    for (usz i = 0; i < self->dim_x; ++i) {
        for (usz j = 0; j < self->dim_y; ++j) {
            for (usz k = 0; k < self->dim_z; ++k) {
                usz index = i * (self->dim_y * self->dim_z) + j * self->dim_z + k;
                fprintf(stderr, "%s%6.3lf%s ",
                        CELL_KIND_CORE == self->kinds[index] ? "\x1b[1m" : "",
                        self->values[index],
                        "\x1b[0m");
            }
            fprintf(stderr, "\n");
        }
        fprintf(stderr, "\n");
    }
}

cell_kind_t mesh_set_cell_kind(const mesh_t* self, usz i, usz j, usz k) {
    if ((i >= STENCIL_ORDER && i < self->dim_x - STENCIL_ORDER) &&
        (j >= STENCIL_ORDER && j < self->dim_y - STENCIL_ORDER) &&
        (k >= STENCIL_ORDER && k < self->dim_z - STENCIL_ORDER)) {
        return CELL_KIND_CORE;
    } else {
        return CELL_KIND_PHANTOM;
    }
}


void mesh_copy_core(mesh_t* dst, const mesh_t* src) {
    assert(dst->dim_x == src->dim_x && dst->dim_y == src->dim_y && dst->dim_z == src->dim_z);

    // Calculate the number of layers, rows, and columns in the core part
    usz core_layers = dst->dim_z - 2 * STENCIL_ORDER;
    usz core_rows = dst->dim_y - 2 * STENCIL_ORDER;
    usz core_columns = dst->dim_x - 2 * STENCIL_ORDER;

    // Iterate only over the core part
    for (usz k = 0; k < core_layers; ++k) {
        for (usz j = 0; j < core_rows; ++j) {
            for (usz i = 0; i < core_columns; ++i) {
                // Calculate the linear index for both source and destination arrays
                usz src_index = ((k + STENCIL_ORDER) * src->dim_y + (j + STENCIL_ORDER)) * src->dim_x + (i + STENCIL_ORDER);
                usz dst_index = ((k + STENCIL_ORDER) * dst->dim_y + (j + STENCIL_ORDER)) * dst->dim_x + (i + STENCIL_ORDER);

                // Copy values assuming all core indices are valid and of type CELL_KIND_CORE
                dst->values[dst_index] = src->values[src_index];
            }
        }
    }
}

