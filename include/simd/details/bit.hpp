#ifndef SIMD_DETAILS_BIT_HPP
#define SIMD_DETAILS_BIT_HPP

#include <bit>
#include <type_traits>

namespace simd::details {

template <class T>
constexpr T set_allbits() {
   if constexpr (std::is_integral_v<T>) {
      using unsigned_value_type = std::make_unsigned_t<T>;
      constexpr unsigned_value_type mask = ~static_cast<unsigned_value_type>(0);
      return static_cast<T>(mask);
   }
   else if constexpr (std::is_same_v<float, T>) {
      return std::bit_cast<float>(details::set_allbits<std::int32_t>());
   }
   else if constexpr (std::is_same_v<double, T>) {
      return std::bit_cast<double>(details::set_allbits<std::int64_t>());
   }
}

int lowest_bit_set(int m) {
   // std:countr_zero?
   return std::countr_zero(static_cast<unsigned int>(m));
}

int lowest_bit_set(unsigned long m) {
   // std:countr_zero?
   return std::countr_zero(m);
}

#if defined(_WIN32) && !defined(__CHAR_BIT__)
#define SIMD_CHAR_BIT 8
#else
#define SIMD_CHAR_BIT __CHAR_BIT__
#endif

int highest_bit_set(int m) {
   // std:countl_zero?
   return (sizeof(int) * SIMD_CHAR_BIT - 1) -
          std::countl_zero(static_cast<unsigned int>(m));
}

int highest_bit_set(unsigned long m) {
   // std:countl_zero?
   return (sizeof(unsigned long) * SIMD_CHAR_BIT - 1) - std::countl_zero(m);
}


}  // namespace simd::details

#endif
