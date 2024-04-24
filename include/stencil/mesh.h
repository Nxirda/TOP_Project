#pragma once

#include "../types.h"

#define STENCIL_ORDER 8UL  

// Enumeration to define the kind of cells in a mesh.
typedef enum {
    CELL_KIND_CORE,   
    CELL_KIND_PHANTOM 
} cell_kind_t;

// Structure for a single cell in the mesh.
typedef struct {
    f64 value;       
    cell_kind_t kind;
} cell_t;

// Enumeration to define the type of mesh.
typedef enum {
    MESH_KIND_CONSTANT, 
    MESH_KIND_INPUT,    // Mesh used for input data.
    MESH_KIND_OUTPUT    // Mesh used for storing output data.
} mesh_kind_t;


typedef struct {
    usz dim_x, dim_y, dim_z;  // Dimensions of the mesh including ghost cells.
    f64* values;              // Array of values for all cells.
    cell_kind_t* kinds;       // Array of kinds for all cells.
    mesh_kind_t kind;         // The kind of mesh.
} mesh_t;

mesh_t mesh_new(usz dim_x, usz dim_y, usz dim_z, mesh_kind_t kind); // Initializes a new mesh.
void mesh_drop(mesh_t* self); // Cleans up and deallocates a mesh.
void mesh_print(const mesh_t* self, const char* name); // Prints mesh details for debugging.
cell_kind_t mesh_set_cell_kind(const mesh_t* self, usz i, usz j, usz k); // Sets the kind of a cell.
void mesh_copy_core(mesh_t* dst, const mesh_t* src); // Copies the core part of one mesh to another.


/// Returns a pointer to the indexed element (includes surrounding ghost cells).
f64* idx(mesh_t* self, usz i, usz j, usz k);

/// Returns a pointer to the indexed element (ignores surrounding ghost cells).
f64* idx_core(mesh_t* self, usz i, usz j, usz k);

/// Returns the value at the indexed element (includes surrounding ghost cells).
f64 idx_const(mesh_t const* self, usz i, usz j, usz k);

/// Returns the value at the indexed element (ignores surrounding ghost cells).
f64 idx_core_const(mesh_t const* self, usz i, usz j, usz k);
