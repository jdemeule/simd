#ifndef SIMD_SIMD_SHAPE_HPP
#define SIMD_SIMD_SHAPE_HPP

#include <cstdint>
#include <type_traits>

namespace simd::details {

enum class simd_shape : std::uint32_t {
   error = 0,

   // clang-format off
   sign_flag = 0x100000,
   // type width flags
   float64x  = 0x010000,
   float32x  = 0x020000,
   
   uint8x     = 0x001000,
   uint16x    = 0x002000,
   uint32x    = 0x004000,
   uint64x    = 0x008000,

   int8x     = sign_flag | uint8x,
   int16x    = sign_flag | uint16x,
   int32x    = sign_flag | uint32x,
   int64x    = sign_flag | uint64x,
   // clang-format on

   float64x1 = float64x | 1,
   float64x2 = float64x | 2,
   float64x4 = float64x | 4,
   float64x8 = float64x | 8,

   float32x2 = float32x | 2,
   float32x4 = float32x | 4,
   float32x8 = float32x | 8,
   float32x16 = float32x | 16,

   int8x8 = int8x | 8,
   int8x16 = int8x | 16,
   int8x32 = int8x | 32,
   int8x64 = int8x | 64,

   uint8x8 = uint8x | 8,
   uint8x16 = uint8x | 16,
   uint8x32 = uint8x | 32,
   uint8x64 = uint8x | 64,

   int16x4 = int16x | 4,
   int16x8 = int16x | 8,
   int16x16 = int16x | 16,
   int16x32 = int16x | 32,

   uint16x4 = uint16x | 4,
   uint16x8 = uint16x | 8,
   uint16x16 = uint16x | 16,
   uint16x32 = uint16x | 32,

   int32x2 = int32x | 2,
   int32x4 = int32x | 4,
   int32x8 = int32x | 8,
   int32x16 = int32x | 16,

   uint32x2 = uint32x | 2,
   uint32x4 = uint32x | 4,
   uint32x8 = uint32x | 8,
   uint32x16 = uint32x | 16,

   int64x1 = int64x | 1,
   int64x2 = int64x | 2,
   int64x4 = int64x | 4,
   int64x8 = int64x | 8,

   uint64x1 = uint64x | 1,
   uint64x2 = uint64x | 2,
   uint64x4 = uint64x | 4,
   uint64x8 = uint64x | 8,
};
constexpr simd_shape operator|(simd_shape lhs, simd_shape rhs) {
   using underlying = std ::underlying_type_t<simd_shape>;
   return static_cast<simd_shape>(static_cast<underlying>(lhs) |
                                  static_cast<underlying>(rhs));
}
constexpr simd_shape operator&(simd_shape lhs, simd_shape rhs) {
   using underlying = std ::underlying_type_t<simd_shape>;
   return static_cast<simd_shape>(static_cast<underlying>(lhs) &
                                  static_cast<underlying>(rhs));
}
constexpr simd_shape operator^(simd_shape lhs, simd_shape rhs) {
   using underlying = std ::underlying_type_t<simd_shape>;
   return static_cast<simd_shape>(static_cast<underlying>(lhs) ^
                                  static_cast<underlying>(rhs));
}
constexpr simd_shape operator~(simd_shape lhs) {
   using underlying = std ::underlying_type_t<simd_shape>;
   return static_cast<simd_shape>(~static_cast<underlying>(lhs));
}
constexpr simd_shape& operator|=(simd_shape& lhs, simd_shape rhs) {
   using underlying = std ::underlying_type_t<simd_shape>;
   lhs = static_cast<simd_shape>(static_cast<underlying>(lhs) |
                                 static_cast<underlying>(rhs));
   return lhs;
}
constexpr simd_shape& operator&=(simd_shape& lhs, simd_shape rhs) {
   using underlying = std ::underlying_type_t<simd_shape>;
   lhs = static_cast<simd_shape>(static_cast<underlying>(lhs) &
                                 static_cast<underlying>(rhs));
   return lhs;
}
constexpr simd_shape& operator^=(simd_shape& lhs, simd_shape rhs) {
   using underlying = std ::underlying_type_t<simd_shape>;
   lhs = static_cast<simd_shape>(static_cast<underlying>(lhs) ^
                                 static_cast<underlying>(rhs));
   return lhs;
};

template <class TVec>
constexpr simd_shape shape_of() noexcept {
   // sizeof(storage) / sizeof(value_type) -> cardinality
   constexpr int N = TVec::size;
   using value_type = typename TVec::value_type;

   if constexpr (std::is_same_v<double, value_type>) {
      return simd_shape::float64x | static_cast<simd_shape>(N);
   }
   else if constexpr (std::is_same_v<float, value_type>) {
      return simd_shape::float32x | static_cast<simd_shape>(N);
   }
   else if constexpr (std::is_integral_v<value_type> && sizeof(value_type) <= 64) {
      if constexpr (std::is_signed_v<value_type>)
         return simd_shape::sign_flag |
                static_cast<simd_shape>(sizeof(value_type) << 12) |
                static_cast<simd_shape>(N);
      else
         return static_cast<simd_shape>(sizeof(value_type) << 12) |
                static_cast<simd_shape>(N);
   }

   return simd_shape::error;
}


template <class T>
struct unsigned_lane {
   using type = std::make_unsigned_t<T>;
};

template <>
struct unsigned_lane<float> {
   using type = std::uint32_t;
};

template <>
struct unsigned_lane<double> {
   using type = std::uint64_t;
};

template <class T>
using unsigned_lane_t = typename unsigned_lane<T>::type;

}  // namespace simd::details



#endif
