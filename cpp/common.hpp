

#ifndef COMMON_HPP
#define COMMON_HPP

#ifndef DEBUG
#define DEBUG 0
#endif
#define debug_printf(fmt, ...) \
do { if (DEBUG) { fprintf(stderr, fmt, __VA_ARGS__); \
fflush(stderr); } } while (0)

#endif //COMMON_HPP
