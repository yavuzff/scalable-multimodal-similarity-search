

#ifndef COMMON_HPP
#define COMMON_HPP

#ifndef NDEBUG
    #define debug_printf(fmt, ...) \
    do { \
    fprintf(stderr, fmt, __VA_ARGS__); \
    fflush(stderr); \
    } while (0)
#else
    #define debug_printf(fmt, ...) ((void)0)
#endif

#endif //COMMON_HPP
