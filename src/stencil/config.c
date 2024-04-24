#include "stencil/config.h"
#include "logging.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline config_t config_default() {
    return (config_t){.dim_x = 100, .dim_y = 100, .dim_z = 100, .niter = 10};
}

config_t config_parse_from_file(char const file_name[static 1]) {
    FILE* cfp = fopen(file_name, "r");
    if (!cfp) {
        warn("Failed to open configuration file %s, using default", file_name);
        return config_default();
    }

    config_t self = config_default();
    char *line_buf = NULL;
    size_t line_buf_size = 0;
    ssize_t line_size;
    size_t line_num = 0;

    while ((line_size = getline(&line_buf, &line_buf_size, cfp)) != -1) {
        line_num++;
        if (line_size > 0 && line_buf[0] == '#') continue;  // Skip comments

        char key[10];
        size_t val;
        if (sscanf(line_buf, "%9s = %zu", key, &val) == 2) {
            if (strcmp("dim_x", key) == 0) self.dim_x = val;
            else if (strcmp("dim_y", key) == 0) self.dim_y = val;
            else if (strcmp("dim_z", key) == 0) self.dim_z = val;
            else if (strcmp("niter", key) == 0) self.niter = val;
            else warn("Unknown configuration key `%s` at line %zu", key, line_num);
        }
    }

    free(line_buf);
    fclose(cfp);
    return self;
}

inline usz config_dim_x(config_t self) {
    return self.dim_x;
}

inline usz config_dim_y(config_t self) {
    return self.dim_y;
}

inline usz config_dim_z(config_t self) {
    return self.dim_z;
}

inline usz config_niter(config_t self) {
    return self.niter;
}

void config_print(const config_t* self) {
    fprintf(stderr,
            "****************************************\n"
            "         STENCIL CONFIGURATION\n"
            "X-axis dimension ................... %zu\n"
            "Y-axis dimension ................... %zu\n"
            "Z-axis dimension ................... %zu\n"
            "Number of iterations ............... %zu\n",
            self->dim_x, self->dim_y, self->dim_z, self->niter);
}
