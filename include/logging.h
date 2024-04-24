#pragma once

#include <stdio.h>
#include <stdlib.h> // Needed for `exit()`

#define info(fmt, ...) \
    do { \
        if (#__VA_ARGS__ == NULL || #__VA_ARGS__[0] == '\0') \
            fprintf(stderr, "\x1b[1;36m[INFO]:\x1b[0m " fmt "\n"); \
        else \
            fprintf(stderr, "\x1b[1;36m[INFO]:\x1b[0m " fmt "\n", ##__VA_ARGS__); \
    } while (0)

#define warn(fmt, ...) \
    do { \
        if (#__VA_ARGS__ == NULL || #__VA_ARGS__[0] == '\0') \
            fprintf(stderr, "\x1b[1;33m[WARNING]:\x1b[0m " fmt "\n"); \
        else \
            fprintf(stderr, "\x1b[1;33m[WARNING]:\x1b[0m " fmt "\n", ##__VA_ARGS__); \
    } while (0)

#define error(fmt, ...) \
    do { \
        if (#__VA_ARGS__ == NULL || #__VA_ARGS__[0] == '\0') \
            fprintf(stderr, "\x1b[1;31m[ERROR]:\x1b[0m " fmt "\n"); \
        else \
            fprintf(stderr, "\x1b[1;31m[ERROR]:\x1b[0m " fmt "\n", ##__VA_ARGS__); \
        exit(-1); \
    } while (0)
