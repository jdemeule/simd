#ifndef SIMD_X86_ARCH_HPP
#define SIMD_X86_ARCH_HPP

#include <simd/details/arch.hpp>

#include <type_traits>

namespace simd::details {

struct arm_isa_tag : public cpu_isa_tag {};
struct neon_isa_tag : public arm_isa_tag {};


enum class arm_isa_level {
   arm = 0,
   neon = 10,
};

#if defined(__ARM_NEON)
inline constexpr auto current_isa = arm_isa_level::neon;
#else
inline constexpr auto current_isa = arm_isa_level::arm;
#endif

constexpr std::size_t native_register_width() {
   return 128;
}

}  // namespace simd::details



#endif
