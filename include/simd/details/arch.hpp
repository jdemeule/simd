#ifndef SIMD_DETAILS_ARCH_HPP
#define SIMD_DETAILS_ARCH_HPP


#include <concepts>

namespace simd::details {

struct cpu_isa_tag {};

template <class T, class Tag>
concept has_isa_tag = std::derived_from<typename T::isa_tag, Tag>;

template <class TVec>
constexpr bool is_full_vec() {
   using Abi = typename TVec::abi_type;
   return Abi::is_full;
}

template <class TVec>
constexpr int full_width() {
   using Abi = typename TVec::abi_type;
   return Abi::width;
}


}  // namespace simd::details


#endif
