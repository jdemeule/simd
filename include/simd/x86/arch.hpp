#ifndef SIMD_X86_ARCH_HPP
#define SIMD_X86_ARCH_HPP

#include <simd/details/arch.hpp>

#include <immintrin.h>


namespace simd::details {

struct x86_isa_tag : public cpu_isa_tag {};
struct sse4_isa_tag : public x86_isa_tag {};
struct avx2_isa_tag : public sse4_isa_tag {};

enum class x86_isa_level {
   x86 = 0,
   sse = 10,
   sse2 = 20,
   sse3 = 30,
   sse4_1 = 41,
   sse4_2 = 42,
   avx = 50,
   avx2 = 60,
   avx512 = 70
};


// Lot of time the shape is enough to use the correct instruction.
// However, something the function is available at upper level of ISA (avx2 when we are on
// __mm128x, or avx512 on __mm256x).
// A simple example is _mm256_min_epi64, missing from avx2 but available on avx512.
// Storing the current isa will allow us this kind of dispatch.

#if defined(__AVX2__)
inline constexpr auto current_isa = x86_isa_level::avx2;
#elif defined(__AVX__)
inline constexpr auto current_isa = x86_isa_level::avx;
#elif defined(__SSE4_2__)
inline constexpr auto current_isa = x86_isa_level::sse4_2;
#elif defined(__SSE4_1__)
inline constexpr auto current_isa = x86_isa_level::sse4_1;
#elif defined(__SSE3__)
inline constexpr auto current_isa = x86_isa_level::sse3;
#elif defined(__SSE2__)
inline constexpr auto current_isa = x86_isa_level::sse2;
#elif defined(__SSE__)
inline constexpr auto current_isa = x86_isa_level::sse;
#else
inline constexpr auto current_isa = x86_isa_level::x86;
#endif

constexpr std::size_t native_register_width() {
   if constexpr (current_isa >= x86_isa_level::avx)
      return 256;
   else if constexpr (current_isa >= x86_isa_level::sse)
      return 128;
   return 0;
}

}  // namespace simd::details



#endif
