#include "stencil/solve.h"

#include <assert.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>

void solve_jacobi(mesh_t *A, const mesh_t *B, mesh_t *C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    // Précalculer les inverses des puissances de 17 pour chaque ordre de stencil
    double powers[STENCIL_ORDER];
    for (int o = 1; o <= STENCIL_ORDER; ++o) {
        powers[o-1] = 1.0 / pow(17.0, o);
    }

    usz i, j, k, o;
    for (i = STENCIL_ORDER; i < A->dim_x - STENCIL_ORDER; ++i) {
        for (j = STENCIL_ORDER; j < A->dim_y - STENCIL_ORDER; ++j) {
            for (k = STENCIL_ORDER; k < A->dim_z - STENCIL_ORDER; k += 4) { 
                __m256d acc = _mm256_setzero_pd();

                // Interaction avec centre
                __m256d self_vals_A = _mm256_loadu_pd(&A->values[i][j][k]);
                __m256d self_vals_B = _mm256_loadu_pd(&B->values[i][j][k]);
                __m256d self_product = _mm256_mul_pd(self_vals_A, self_vals_B);
                acc = _mm256_add_pd(acc, self_product);

                for (o = 1; o <= STENCIL_ORDER; ++o) {
                    double factor = powers[o-1];
                    __m256d factor_vec = _mm256_set1_pd(factor);

                    // Voisins horizontaux et verticaux
                    __m256d neighbors_A = _mm256_set_pd(
                        A->values[i+o][j][k], A->values[i-o][j][k],
                        A->values[i][j+o][k], A->values[i][j-o][k]
                    );
                    __m256d neighbors_B = _mm256_set_pd(
                        B->values[i+o][j][k], B->values[i-o][j][k],
                        B->values[i][j+o][k], B->values[i][j-o][k]
                    );

                    // Voisins en profondeur
                    __m256d neighbors_k_A = _mm256_set_pd(
                        A->values[i][j][k+o], A->values[i][j][k-o],
                        0, 0 
                    );
                    __m256d neighbors_k_B = _mm256_set_pd(
                        B->values[i][j][k+o], B->values[i][j][k-o],
                        0, 0
                    );
                    __m256d neighbor_products = _mm256_mul_pd(_mm256_add_pd(neighbors_A, neighbors_k_A), _mm256_add_pd(neighbors_B, neighbors_k_B));
                    __m256d contributions = _mm256_mul_pd(neighbor_products, factor_vec);
                    acc = _mm256_add_pd(acc, contributions);
                }
                _mm256_storeu_pd(&C->values[i][j][k], acc);
            }
        }
    }
    mesh_copy_core(A, C);
}


/*
void solve_jacobi(mesh_t *A, const mesh_t *B, mesh_t *C) {
    assert(A->dim_x == B->dim_x && B->dim_x == C->dim_x);
    assert(A->dim_y == B->dim_y && B->dim_y == C->dim_y);
    assert(A->dim_z == B->dim_z && B->dim_z == C->dim_z);

    // Précalculer les inverses des puissances de 17 pour chaque ordre de stencil
    double powers[STENCIL_ORDER];
    for (int o = 1; o <= STENCIL_ORDER; ++o) {
        powers[o-1] = 1.0 / pow(17.0, o);
    }

    usz i, j, k, o;
    // Parcourir tous les éléments du maillage, en évitant les bords
    for (i = STENCIL_ORDER; i < A->dim_x - STENCIL_ORDER; ++i) {
        for (j = STENCIL_ORDER; j < A->dim_y - STENCIL_ORDER; ++j) {
            for (k = STENCIL_ORDER; k < A->dim_z - STENCIL_ORDER; ++k) {
                // Accumulateur pour le résultat final pour C[i][j][k]
                __m256d acc = _mm256_setzero_pd();

                // Interaction avec la cellule elle-même
                __m256d self_val_A = _mm256_broadcast_sd(&A->values[i][j][k]);
                __m256d self_val_B = _mm256_broadcast_sd(&B->values[i][j][k]);
                __m256d product = _mm256_mul_pd(self_val_A, self_val_B);
                acc = _mm256_add_pd(acc, product);

                // Interactions avec les voisins
                for (o = 1; o <= STENCIL_ORDER; ++o) {
                    double factor = powers[o-1];

                    // Charger les valeurs des voisins
                    __m256d neighbor_vals_A = _mm256_set_pd(
                        A->values[i+o][j][k], A->values[i-o][j][k],
                        A->values[i][j+o][k], A->values[i][j-o][k]
                    );
                    __m256d neighbor_vals_B = _mm256_set_pd(
                        B->values[i+o][j][k], B->values[i-o][j][k],
                        B->values[i][j+o][k], B->values[i][j-o][k]
                    );

                    // Calculer les produits et les ajouter à l'accumulateur
                    __m256d factor_vec = _mm256_broadcast_sd(&factor);
                    __m256d neighbor_products = _mm256_mul_pd(neighbor_vals_A, neighbor_vals_B);
                    __m256d contributions = _mm256_mul_pd(neighbor_products, factor_vec);
                    acc = _mm256_add_pd(acc, contributions);
                }

                // Stocker le résultat accumulé dans C
                double result;
                _mm256_store_pd(&result, acc);
                C->values[i][j][k] = result;
            }
        }
    }
    mesh_copy_core(A, C);
}*/

