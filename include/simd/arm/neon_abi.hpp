#ifndef SIMD_ARM_NEON_ABI_HPP
#define SIMD_ARM_NEON_ABI_HPP


#include <simd/arm/arch.hpp>
#include <simd/details/bit.hpp>
#include <simd/details/simd_fwd.hpp>
#include <simd/details/simd_shape.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

#include <arm_neon.h>


namespace simd {


namespace details {



// template <class TVec>
// constexpr bool is_full_vec() {
//    using Abi = typename TVec::abi_type;
//    return Abi::is_full;
// }

// template <class TVec>
// constexpr int full_width() {
//    using Abi = typename TVec::abi_type;
//    return Abi::width;
// }

struct arm_neon_isa {
   template <class TVec>
   static TVec make(typename TVec::value_type value) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return vmovq_n_u8(static_cast<std::int8_t>(value));
      else if constexpr (s == simd_shape::int16x8)
         return vmovq_n_u16(value);
      else if constexpr (s == simd_shape::int32x4)
         return vmovq_n_u32(value);
      else if constexpr (s == simd_shape::int64x2)
         return vmovq_n_u64(value);
      else if constexpr (s == simd_shape::uint8x16)
         return vmovq_n_s8(static_cast<std::uint8_t>(value));
      else if constexpr (s == simd_shape::uint16x8)
         return vmovq_n_s16(value);
      else if constexpr (s == simd_shape::uint32x4)
         return vmovq_n_s32(value);
      else if constexpr (s == simd_shape::uint64x2)
         return vmovq_n_s64(value);
      else if constexpr (s == simd_shape::float32x4)
         return vmovq_n_f32(value);
      else if constexpr (s == simd_shape::float64x2)
         return vmovq_n_f64(value);
   }

   template <class TVec, class... Ts>
   static TVec make_vs(Ts... values) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return int8x16_t{static_cast<std::int8_t>(values)...};
      else if constexpr (s == simd_shape::int16x8)
         return int16x8_t{values...};
      else if constexpr (s == simd_shape::int32x4)
         return int32x4_t{values...};
      else if constexpr (s == simd_shape::int64x2)
         return int64x2_t{values...};
      else if constexpr (s == simd_shape::uint8x16)
         return uint8x16_t{static_cast<std::uint8_t>(values)...};
      else if constexpr (s == simd_shape::uint16x8)
         return uint16x8_t{values...};
      else if constexpr (s == simd_shape::uint32x4)
         return uint32x4_t{values...};
      else if constexpr (s == simd_shape::uint64x2)
         return uint64x2_t{values...};
      else if constexpr (s == simd_shape::float32x4)
         return float32x4_t{values...};
      else if constexpr (s == simd_shape::float64x2)
         return float64x2_t{values...};
   }

   template <class TVec, class G, std::size_t... Is>
   static TVec make_gen(G&& gen, std::index_sequence<Is...>) {
      return make_vs<TVec>(gen(std::integral_constant<int, Is>{})...);
   }

   template <class TVec, class G>
   static TVec generate(G&& gen) {
      return make_gen<TVec>(std::forward<G>(gen), std::make_index_sequence<TVec::size>{});
   }

   template <class TVec, class UVec>
   static TVec convert(const UVec& x) {
      return generate<TVec>([&](auto i) { return x[i]; });
   }



   template <class TVec, class TMask, class It>
   static auto load(It it, const TMask& m) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::float32x4)
         return simd_select(m, TVec(vld1q_f32(std::to_address(it))), TVec(0));
      else if constexpr (s == simd_shape::float64x2)
         return simd_select(m, TVec(vld1q_f64(std::to_address(it))), TVec(0));
      else if constexpr (s == simd_shape::int8x16)
         return simd_select(m, TVec(vld1q_s8((const std::int8_t*)std::to_address(it))),
                            TVec(0));
      else if constexpr (s == simd_shape::int16x8)
         return simd_select(m, TVec(vld1q_s16(std::to_address(it))), TVec(0));
      else if constexpr (s == simd_shape::int32x4)
         return simd_select(m, TVec(vld1q_s32(std::to_address(it))), TVec(0));
      else if constexpr (s == simd_shape::int64x2)
         return simd_select(m, TVec(vld1q_s64(std::to_address(it))), TVec(0));
      else if constexpr (s == simd_shape::uint8x16)
         return simd_select(m, TVec(vld1q_u8((const std::uint8_t*)std::to_address(it))),
                            TVec(0));
      else if constexpr (s == simd_shape::uint16x8)
         return simd_select(m, TVec(vld1q_u16((const std::uint16_t*)std::to_address(it))),
                            TVec(0));
      else if constexpr (s == simd_shape::uint32x4)
         return simd_select(m, TVec(vld1q_u32(std::to_address(it))), TVec(0));
      else if constexpr (s == simd_shape::uint64x2)
         return simd_select(m, TVec(vld1q_u64(std::to_address(it))), TVec(0));
   }

   template <class TVec, class It>
   static auto load(It it) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (is_full_vec<TVec>()) {
         if constexpr (s == simd_shape::float32x4)
            return vld1q_f32(std::to_address(it));
         else if constexpr (s == simd_shape::float64x2)
            return vld1q_f64(std::to_address(it));
         else if constexpr (s == simd_shape::int8x16)
            return vld1q_s8(reinterpret_cast<const std::int8_t*>(std::to_address(it)));
         else if constexpr (s == simd_shape::int16x8)
            return vld1q_s16(std::to_address(it));
         else if constexpr (s == simd_shape::int32x4)
            return vld1q_s32(std::to_address(it));
         else if constexpr (s == simd_shape::int64x2)
            return vld1q_s64(std::to_address(it));
         else if constexpr (s == simd_shape::uint8x16)
            return vld1q_u8(reinterpret_cast<const std::uint8_t*>(std::to_address(it)));
         else if constexpr (s == simd_shape::uint16x8)
            return vld1q_u16(reinterpret_cast<const std::uint16_t*>(std::to_address(it)));
         else if constexpr (s == simd_shape::uint32x4)
            return vld1q_u32(std::to_address(it));
         else if constexpr (s == simd_shape::uint64x2)
            return vld1q_u64(std::to_address(it));
      }
      else {
         return load(it, [](auto i) { return i < TVec::size; });
      }
   }



   template <class TVec, class TMask, class Out>
   static void store(Out out, const TVec& self, const TMask& m) {
      // constexpr simd_shape s = shape_of<TVec>();

      auto fallback_store = [](auto out, const auto& self, const auto& m) {
         // memcpy or store?
         typename TVec::value_type data[TVec::size];
         std::memcpy(&data, &self, sizeof(data));
         for (std::size_t i = 0; i < TVec::size; ++i, ++out) {
            if (m[i])
               *out = data[i];
         }
      };
      return fallback_store(out, self, m);
   }

   template <class TVec, class Out>
   static void store(Out out, const TVec& self) {
      constexpr simd_shape s = shape_of<TVec>();
      // constexpr int width = full_width<TVec>();

      if constexpr (is_full_vec<TVec>()) {
         if constexpr (s == simd_shape::float32x4)
            return vst1q_f32(std::to_address(out), self);
         else if constexpr (s == simd_shape::float64x2)
            return vst1q_f64(std::to_address(out), self);
         else if constexpr (s == simd_shape::int8x16)
            return vst1q_s8(std::to_address(out), self);
         else if constexpr (s == simd_shape::int16x8)
            return vst1q_s16(std::to_address(out), self);
         else if constexpr (s == simd_shape::int32x4)
            return vst1q_s32(std::to_address(out), self);
         else if constexpr (s == simd_shape::int64x2)
            return vst1q_s64(std::to_address(out), self);
         else if constexpr (s == simd_shape::uint8x16)
            return vst1q_u8(reinterpret_cast<std::uint8_t*>(std::to_address(out)), self);
         else if constexpr (s == simd_shape::uint16x8)
            return vst1q_u16(reinterpret_cast<std::uint16_t*>(std::to_address(out)),
                             self);
         else if constexpr (s == simd_shape::uint32x4)
            return vst1q_u32(std::to_address(out), self);
         else if constexpr (s == simd_shape::uint64x2)
            return vst1q_u64(std::to_address(out), self);
      }
      else {
         return store(out, self, [](auto i) { return i < TVec::size; });
      }
   }

   template <class TVec, class Op>
   static TVec apply_op(const TVec& lhs, const TVec& rhs, Op&& op) {
      using value_type = typename TVec::value_type;
      std::array<value_type, TVec::size> a;
      std::array<value_type, TVec::size> b;
      lhs.copy_to(a.begin());
      rhs.copy_to(b.begin());

      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = static_cast<value_type>(op(a[i], b[i]));
      }
      return TVec{buffer.begin()};
   }


   template <class TVec>
   static TVec add(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return vaddq_s8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return vaddq_s16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return vaddq_s32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return vaddq_s64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return vaddq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return vaddq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return vaddq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return vaddq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4)
         return vaddq_f32(lhs, rhs);
      else if constexpr (s == simd_shape::float64x2)
         return vaddq_f64(lhs, rhs);
   }

   template <class TVec>
   static TVec sub(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return vsubq_s8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return vsubq_s16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return vsubq_s32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return vsubq_s64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return vsubq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return vsubq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return vsubq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return vsubq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4)
         return vsubq_f32(lhs, rhs);
      else if constexpr (s == simd_shape::float64x2)
         return vsubq_f64(lhs, rhs);
      else
         static_assert(std::is_same_v<void, TVec>, "Unsupported shape");
   }

   template <class TVec>
   static TVec mul(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return vmulq_s8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return vmulq_s16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return vmulq_s32(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return vmulq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return vmulq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return vmulq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4)
         return vmulq_f32(lhs, rhs);
      else if constexpr (s == simd_shape::float64x2)
         return vmulq_f64(lhs, rhs);
      else
         return apply_op(lhs, rhs, [](auto a, auto b) { return a * b; });
   }

   template <class TVec>
   static TVec div(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::float32x4) {
         return vdivq_f32(lhs, rhs);
      }
      else if constexpr (s == simd_shape::float64x2) {
         return vdivq_f64(lhs, rhs);
      }
      else {
         return apply_op(lhs, rhs, [](auto a, auto b) { return a / b; });
      }
   }

   template <class TVec>
   static TVec rem(const TVec& lhs, const TVec& rhs) {
      return apply_op(lhs, rhs, [](auto a, auto b) { return a % b; });
   }

   template <class TVec>
   static TVec min(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return vminq_s8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return vminq_s16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return vminq_s32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return vminq_s64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return vminq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return vminq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return vminq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return vminq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4)
         return vminq_f32(lhs, rhs);
      else if constexpr (s == simd_shape::float64x2)
         return vminq_f64(lhs, rhs);
   }

   template <class TVec>
   static TVec max(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return vmaxq_s8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return vmaxq_s16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return vmaxq_s32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return vmaxq_s64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return vmaxq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return vmaxq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return vmaxq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return vmaxq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4)
         return vmaxq_f32(lhs, rhs);
      else if constexpr (s == simd_shape::float64x2)
         return vmaxq_f64(lhs, rhs);
   }

   template <class TVec>
   static TVec unary_minus(const TVec& self) {
      return TVec{0} - self;
   }

   template <class TVec>
   static TVec bit_not(const TVec& self) {
      using value_type = typename TVec::value_type;
      constexpr auto all_bits_set = details::set_allbits<value_type>();
      return bit_xor(TVec{static_cast<value_type>(all_bits_set)}, self);
   }

   template <class TVec>
   static TVec bit_and(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return vandq_s8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return vandq_s16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return vandq_s32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return vandq_s64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return vandq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return vandq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return vandq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return vandq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4) {
         auto lhs_u32 = vreinterpretq_u32_f32(lhs);
         auto rhs_u32 = vreinterpretq_u32_f32(rhs);
         return vreinterpretq_f32_u32(vandq_u32(lhs_u32, rhs_u32));
      }
      else if constexpr (s == simd_shape::float64x2) {
         auto lhs_u64 = vreinterpretq_u64_f64(lhs);
         auto rhs_u64 = vreinterpretq_u64_f64(rhs);
         return vreinterpretq_f64_u64(vandq_u64(lhs_u64, rhs_u64));
      }
   }

   template <class TVec>
   static TVec bit_or(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return vorrq_s8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return vorrq_s16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return vorrq_s32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return vorrq_s64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return vorrq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return vorrq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return vorrq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return vorrq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4) {
         auto lhs_u32 = vreinterpretq_u32_f32(lhs);
         auto rhs_u32 = vreinterpretq_u32_f32(rhs);
         return vreinterpretq_f32_u32(vorrq_u32(lhs_u32, rhs_u32));
      }
      else if constexpr (s == simd_shape::float64x2) {
         auto lhs_u64 = vreinterpretq_u64_f64(lhs);
         auto rhs_u64 = vreinterpretq_u64_f64(rhs);
         return vreinterpretq_f64_u64(vorrq_u64(lhs_u64, rhs_u64));
      }
   }

   template <class TVec>
   static TVec bit_xor(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return veorq_s8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return veorq_s16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return veorq_s32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return veorq_s64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return veorq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return veorq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return veorq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return veorq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4) {
         auto lhs_u32 = vreinterpretq_u32_f32(lhs);
         auto rhs_u32 = vreinterpretq_u32_f32(rhs);
         return vreinterpretq_f32_u32(veorq_u32(lhs_u32, rhs_u32));
      }
      else if constexpr (s == simd_shape::float64x2) {
         auto lhs_u64 = vreinterpretq_u64_f64(lhs);
         auto rhs_u64 = vreinterpretq_u64_f64(rhs);
         return vreinterpretq_f64_u64(veorq_u64(lhs_u64, rhs_u64));
      }
   }

   template <class TVec>
   static TVec bit_shift_left(const TVec& lhs, int count) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return vshlq_s8(lhs, TVec(count));
      else if constexpr (s == simd_shape::int16x8)
         return vshlq_s16(lhs, TVec(count));
      else if constexpr (s == simd_shape::int32x4)
         return vshlq_s32(lhs, TVec(count));
      else if constexpr (s == simd_shape::int64x2)
         return vshlq_s64(lhs, TVec(count));
      else if constexpr (s == simd_shape::uint8x16)
         return vshlq_u8(lhs, TVec(count));
      else if constexpr (s == simd_shape::uint16x8)
         return vshlq_u16(lhs, TVec(count));
      else if constexpr (s == simd_shape::uint32x4)
         return vshlq_u32(lhs, TVec(count));
      else if constexpr (s == simd_shape::uint64x2)
         return vshlq_u64(lhs, TVec(count));
   }

   template <class TVec>
   static TVec bit_shift_right(const TVec& lhs, int count) {
      return bit_shift_left(lhs, -count);
   }

   template <class TVec>
   static auto cmp_eq(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return vceqq_s8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return vceqq_s16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return vceqq_s32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return vceqq_s64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return vceqq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return vceqq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return vceqq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return vceqq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4)
         return vceqq_f32(lhs, rhs);
      else if constexpr (s == simd_shape::float64x2)
         return vceqq_f64(lhs, rhs);
   }

   template <class TVec>
   static auto cmp_lt(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return vcltq_s8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return vcltq_s16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return vcltq_s32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return vcltq_s64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return vcltq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return vcltq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return vcltq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return vcltq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4)
         return vcltq_f32(lhs, rhs);
      else if constexpr (s == simd_shape::float64x2)
         return vcltq_f64(lhs, rhs);
   }

   template <class TVec, class TVecMask>
   static auto blend(const TVecMask& m, const TVec& rhs, const TVec& lhs) {
      // Follow Intel blend convention here.
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::float32x4)
         return vbslq_f32(m, lhs, rhs);
      else if constexpr (s == simd_shape::float64x2)
         return vbslq_f64(m, lhs, rhs);
      else if constexpr (s == simd_shape::int8x16)
         return vbslq_s8(m, lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return vbslq_s16(m, lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return vbslq_s32(m, lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return vbslq_s64(m, lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return vbslq_u8(m, lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return vbslq_u16(m, lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return vbslq_u32(m, lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return vbslq_u64(m, lhs, rhs);
   }

   template <class TVec>
   static auto get_lane(const TVec& self, std::size_t index) {
      typename TVec::value_type data[TVec::size];
      std::memcpy(&data, &self, sizeof(data));
      return data[index];
   }

   template <class TVec>
   static void set_lane(typename TVec::value_type value, TVec& self, std::size_t index) {
      // memcpy or store?
      typename TVec::value_type data[TVec::size];
      std::memcpy(&data, &self, sizeof(data));
      data[index] = value;
      // memcpy or load?
      std::memcpy(&self, &data, sizeof(data));
   }

   template <class TMask>
   static TMask make_mask(bool value) {
      constexpr simd_shape s = shape_of<typename TMask::abi_type>();

      if constexpr (s == simd_shape::int8x16)
         return vmovq_n_u8(value ? details::set_allbits<std::uint8_t>() : 0);
      else if constexpr (s == simd_shape::int16x8)
         return vmovq_n_u16(value ? details::set_allbits<std::uint16_t>() : 0);
      else if constexpr (s == simd_shape::int32x4)
         return vmovq_n_u32(value ? details::set_allbits<std::uint32_t>() : 0);
      else if constexpr (s == simd_shape::int64x2)
         return vmovq_n_u64(value ? details::set_allbits<std::uint64_t>() : 0);
      else if constexpr (s == simd_shape::uint8x16)
         return vmovq_n_u8(value ? details::set_allbits<std::uint8_t>() : 0);
      else if constexpr (s == simd_shape::uint16x8)
         return vmovq_n_u16(value ? details::set_allbits<std::uint16_t>() : 0);
      else if constexpr (s == simd_shape::uint32x4)
         return vmovq_n_u32(value ? details::set_allbits<std::uint32_t>() : 0);
      else if constexpr (s == simd_shape::uint64x2)
         return vmovq_n_u64(value ? details::set_allbits<std::uint64_t>() : 0);
      else if constexpr (s == simd_shape::float32x4)
         return vmovq_n_u32(value ? details::set_allbits<std::uint32_t>() : 0);
      else if constexpr (s == simd_shape::float64x2)
         return vmovq_n_u64(value ? details::set_allbits<std::uint64_t>() : 0);
   }

   template <class TMask, class... Ts>
   static TMask make_mask_vs(Ts... values) {
      constexpr simd_shape s = shape_of<typename TMask::abi_type>();

      if constexpr (s == simd_shape::int8x16)
         return uint8x16_t{static_cast<std::uint8_t>(values)...};
      else if constexpr (s == simd_shape::int16x8)
         return uint16x8_t{static_cast<std::uint16_t>(values)...};
      else if constexpr (s == simd_shape::int32x4)
         return uint32x4_t{static_cast<std::uint32_t>(values)...};
      else if constexpr (s == simd_shape::int64x2)
         return uint64x2_t{static_cast<std::uint64_t>(values)...};
      else if constexpr (s == simd_shape::uint8x16)
         return uint8x16_t{static_cast<std::uint8_t>(values)...};
      else if constexpr (s == simd_shape::uint16x8)
         return uint16x8_t{static_cast<std::uint16_t>(values)...};
      else if constexpr (s == simd_shape::uint32x4)
         return uint32x4_t{static_cast<std::uint32_t>(values)...};
      else if constexpr (s == simd_shape::uint64x2)
         return uint64x2_t{static_cast<std::uint64_t>(values)...};
      else if constexpr (s == simd_shape::float32x4)
         return uint32x4_t{static_cast<std::uint32_t>(values)...};
      else if constexpr (s == simd_shape::float64x2)
         return uint64x2_t{static_cast<std::uint64_t>(values)...};
   }

   template <class TMask, class G, std::size_t... Is>
   static TMask make_gen_mask(G&& gen, std::index_sequence<Is...>) {
      using T = typename TMask::abi_type::mask_lane_type;
      constexpr T one = details::set_allbits<T>();
      constexpr T zero = 0;
      return make_mask_vs<TMask>(
         (static_cast<bool>(gen(std::integral_constant<int, Is>{})) ? one : zero)...);
   }

   template <class TMask, class G>
   static TMask generate_mask(G&& gen) {
      // TODO: issue if TVec::size < native size :/ (masking?)
      return make_gen_mask<TMask>(std::forward<G>(gen),
                                  std::make_index_sequence<TMask::size>{});
   }

   template <class TMask, class UMask>
   static TMask convert_mask(const UMask& x) {
      return generate_mask<TMask>([&](auto i) { return x[i]; });
   }

   template <class TMask, class It>
   static TMask load_mask(It it, const TMask& m) {
      constexpr simd_shape s = shape_of<typename TMask::abi_type>();

      typename TMask::mask_lane_type buffer[TMask::size];  // full
      for (int i = 0; i < TMask::size; ++i, ++it)
         buffer[i] = m[i] ? (*it ? details::set_allbits<std::int8_t>() : 0) : 0;

      if constexpr (s == simd_shape::float32x4)
         return vld1q_u32(buffer);
      else if constexpr (s == simd_shape::float64x2)
         return vld1q_u64(buffer);
      else if constexpr (s == simd_shape::int8x16)
         return vld1q_u8(buffer);
      else if constexpr (s == simd_shape::int16x8)
         return vld1q_u16(buffer);
      else if constexpr (s == simd_shape::int32x4)
         return vld1q_u32(buffer);
      else if constexpr (s == simd_shape::int64x2)
         return vld1q_u64(buffer);
      else if constexpr (s == simd_shape::uint8x16)
         return vld1q_u8(buffer);
      else if constexpr (s == simd_shape::uint16x8)
         return vld1q_u16(buffer);
      else if constexpr (s == simd_shape::uint32x4)
         return vld1q_u32(buffer);
      else if constexpr (s == simd_shape::uint64x2)
         return vld1q_u64(buffer);
   }

   template <class TMask, class It>
   static TMask load_mask(It it) {
      constexpr simd_shape s = shape_of<typename TMask::abi_type>();

      if constexpr (is_full_vec<TMask>()) {
         typename TMask::mask_lane_type data[TMask::size];
         for (int i = 0; i < TMask::size; ++i, ++it)
            data[i] = *it ? details::set_allbits<std::int8_t>() : 0;

         if constexpr (s == simd_shape::float32x4)
            return vld1q_u32(data);
         else if constexpr (s == simd_shape::float64x2)
            return vld1q_u64(data);
         else if constexpr (s == simd_shape::int8x16)
            return vld1q_u8(data);
         else if constexpr (s == simd_shape::int16x8)
            return vld1q_u16(data);
         else if constexpr (s == simd_shape::int32x4)
            return vld1q_u32(data);
         else if constexpr (s == simd_shape::int64x2)
            return vld1q_u64(data);
         else if constexpr (s == simd_shape::uint8x16)
            return vld1q_u8(data);
         else if constexpr (s == simd_shape::uint16x8)
            return vld1q_u16(data);
         else if constexpr (s == simd_shape::uint32x4)
            return vld1q_u32(data);
         else if constexpr (s == simd_shape::uint64x2)
            return vld1q_u64(data);
      }
      else {
         return load_mask(it, [](auto i) { return i < TMask::size; });
      }
   }


   template <class TMask, class Out>
   static void store_mask(Out out, const TMask& self, const TMask& m) {
      typename TMask::mask_lane_type data[TMask::size];
      std::memcpy(&data, &self, sizeof(data));
      for (std::size_t i = 0; i < TMask::size; ++i, ++out) {
         if (m[i])
            *out = static_cast<bool>(m[i]);
      }
   }

   template <class TMask, class Out>
   static void store_mask(Out out, const TMask& self) {
      if constexpr (is_full_vec<TMask>()) {
         // memcpy or store?
         typename TMask::mask_lane_type data[TMask::size];
         std::memcpy(&data, &self, sizeof(data));
         for (std::size_t i = 0; i < TMask::size; ++i, ++out) {
            *out = static_cast<bool>(data[i]);
         }
      }
      else {
         return store_mask(out, self, [](auto i) { return i < TMask::size; });
      }
   }

   template <class TMask>
   static TMask logical_eq(const TMask& lhs, const TMask& rhs) {
      constexpr simd_shape s = shape_of<typename TMask::abi_type>();

      if constexpr (s == simd_shape::int8x16)
         return vceqq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return vceqq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return vceqq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return vceqq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return vceqq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return vceqq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return vceqq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return vceqq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4)
         return vceqq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::float64x2)
         return vceqq_u64(lhs, rhs);
   }

   template <class TMask>
   static bool mask_get_lane(const TMask& self, std::size_t index) {
      typename TMask::mask_lane_type data[TMask::size];
      std::memcpy(&data, &self, sizeof(data));
      return data[index] == details::set_allbits<typename TMask::mask_lane_type>();
   }

   template <class TMask>
   static void mask_set_lane(bool value, TMask& self, std::size_t index) {
      typename TMask::mask_lane_type data[TMask::size];
      std::memcpy(&data, &self, sizeof(data));
      data[index] = value ? details::set_allbits<typename TMask::mask_lane_type>() : 0;
      std::memcpy(&self, &data, sizeof(data));
   }

   template <class TMask>
   static TMask mask_bit_not(const TMask& self) {
      return mask_bit_xor(TMask{true}, self);
   }

   template <class TMask>
   static TMask mask_bit_and(const TMask& lhs, const TMask& rhs) {
      constexpr simd_shape s = shape_of<typename TMask::abi_type>();

      if constexpr (s == simd_shape::int8x16)
         return vandq_s8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return vandq_s16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return vandq_s32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return vandq_s64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return vandq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return vandq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return vandq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return vandq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4) {
         auto lhs_u32 = vreinterpretq_u32_f32(lhs);
         auto rhs_u32 = vreinterpretq_u32_f32(rhs);
         return vreinterpretq_f32_u32(vandq_u32(lhs_u32, rhs_u32));
      }
      else if constexpr (s == simd_shape::float64x2) {
         auto lhs_u64 = vreinterpretq_u64_f64(lhs);
         auto rhs_u64 = vreinterpretq_u64_f64(rhs);
         return vreinterpretq_f64_u64(vandq_u64(lhs_u64, rhs_u64));
      }
   }

   template <class TMask>
   static TMask mask_bit_or(const TMask& lhs, const TMask& rhs) {
      constexpr simd_shape s = shape_of<typename TMask::abi_type>();

      if constexpr (s == simd_shape::int8x16)
         return vorrq_s8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return vorrq_s16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return vorrq_s32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return vorrq_s64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return vorrq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return vorrq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return vorrq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return vorrq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4) {
         auto lhs_u32 = vreinterpretq_u32_f32(lhs);
         auto rhs_u32 = vreinterpretq_u32_f32(rhs);
         return vreinterpretq_f32_u32(vorrq_u32(lhs_u32, rhs_u32));
      }
      else if constexpr (s == simd_shape::float64x2) {
         auto lhs_u64 = vreinterpretq_u64_f64(lhs);
         auto rhs_u64 = vreinterpretq_u64_f64(rhs);
         return vreinterpretq_f64_u64(vorrq_u64(lhs_u64, rhs_u64));
      }
   }

   template <class TMask>
   static TMask mask_bit_xor(const TMask& lhs, const TMask& rhs) {
      constexpr simd_shape s = shape_of<typename TMask::abi_type>();

      if constexpr (s == simd_shape::int8x16)
         return veorq_s8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return veorq_s16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return veorq_s32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return veorq_s64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return veorq_u8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return veorq_u16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return veorq_u32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return veorq_u64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4) {
         auto lhs_u32 = vreinterpretq_u32_f32(lhs);
         auto rhs_u32 = vreinterpretq_u32_f32(rhs);
         return vreinterpretq_f32_u32(veorq_u32(lhs_u32, rhs_u32));
      }
      else if constexpr (s == simd_shape::float64x2) {
         auto lhs_u64 = vreinterpretq_u64_f64(lhs);
         auto rhs_u64 = vreinterpretq_u64_f64(rhs);
         return vreinterpretq_f64_u64(veorq_u64(lhs_u64, rhs_u64));
      }
   }


   template <class TMask>
   static std::uint64_t reduce_mask_raw(const TMask& k) {
      const uint16x8_t m = vreinterpretq_u16_u8(k);
      const uint8x8_t res = vshrn_n_u16(m, 4);
      return vget_lane_u64(vreinterpret_u64_u8(res), 0);
   }

   template <class TMask>
   static std::uint64_t reduce_mask(const TMask& k) {
      std::uint64_t m = reduce_mask_raw(k);

      constexpr simd_shape s = shape_of<typename TMask::abi_type>();
      if constexpr (s == simd_shape::int8x16 || s == simd_shape::uint8x16)
         m &= 0x11'11'11'11'11'11'11'11;
      else if constexpr (s == simd_shape::int16x8 || s == simd_shape::uint16x8)
         m &= 0x01'01'01'01'01'01'01'01;
      else if constexpr (s == simd_shape::int32x4 || s == simd_shape::uint32x4 ||
                         s == simd_shape::float32x4)
         m &= 0x00'01'00'01'00'01'00'01;
      else if constexpr (s == simd_shape::int64x2 || s == simd_shape::uint64x2 ||
                         s == simd_shape::float64x2)
         m &= 0x00'00'00'01'00'00'00'01;
      return m;
   }

   template <class TMask>
   static bool all_of(const TMask& k) {
      std::uint64_t m = reduce_mask_raw(k);
      return m == 0xffffffffffffffff;
   }

   template <class TMask>
   static bool is_zero(const TMask& k) {
      std::uint64_t m = reduce_mask_raw(k);
      return m == 0;
   }

   template <class TMask>
   static int reduce_count(const TMask& k) {
      std::uint64_t m = reduce_mask(k);
      return std::popcount(m);
   }

   template <class TMask>
   static constexpr auto index_coef() {
      constexpr simd_shape s = shape_of<typename TMask::abi_type>();
      if constexpr (s == simd_shape::int8x16 || s == simd_shape::uint8x16)
         return 4;
      else if constexpr (s == simd_shape::int16x8 || s == simd_shape::uint16x8)
         return 8;
      else if constexpr (s == simd_shape::int32x4 || s == simd_shape::uint32x4 ||
                         s == simd_shape::float32x4)
         return 16;
      else if constexpr (s == simd_shape::int64x2 || s == simd_shape::uint64x2 ||
                         s == simd_shape::float64x2)
         return 32;
   }

   template <class TMask>
   static int reduce_min_index(const TMask& k) {
      auto m = reduce_mask(k);
      constexpr auto c = index_coef<TMask>();
      return lowest_bit_set(m) / c;
   }

   template <class TMask>
   static int reduce_max_index(const TMask& k) {
      auto m = reduce_mask(k);
      constexpr auto c = index_coef<TMask>();
      return highest_bit_set(m) / c;
   }

   template <class TVec>
   static TVec byteswap(const TVec& x) {
      constexpr simd_shape s = shape_of<TVec>();
      using SimdU8 = rebind_t<std::uint8_t, TVec>;
      if constexpr (s == simd_shape::uint16x8) {
         return std::bit_cast<TVec>(vrev16q_u8(std::bit_cast<SimdU8>(x)));
      }
      else if constexpr (s == simd_shape::uint32x4) {
         return std::bit_cast<TVec>(vrev32q_u8(std::bit_cast<SimdU8>(x)));
      }
      else if constexpr (s == simd_shape::uint64x2) {
         return std::bit_cast<TVec>(vrev64q_u8(std::bit_cast<SimdU8>(x)));
      }
   }

   template <std::size_t N, class MaskGen>
   static constexpr auto make_compress_mask(MaskGen g) {
      std::array<int, N> pattern{-1};

      int index = 0;
      for_template(
         [&](auto i) {
            if (g(i)) {
               pattern[index++] = i;
            }
         },
         std::make_index_sequence<N>{});

      return pattern;
   }

   template <class TVec, class MaskGen>
   static TVec compress(const TVec& x, MaskGen g) {
      constexpr auto pattern = make_compress_mask<TVec::size>(g);

      using Mask = typename TVec::mask_type;
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16 || s == simd_shape::uint8x16) {
         TVec perm{[&pattern](auto i) { return pattern[i]; }};
         return vqtbl1q_s8(x, perm);
      }
   }

   template <std::size_t N, class MaskGen>
   static constexpr auto make_expand_mask(MaskGen g) {
      std::array<int, N> pattern{-1};
      std::fill_n(pattern.begin(), N, -1);

      int index = 0;
      for_template(
         [&](auto i) {
            if (g(i)) {
               pattern[i] = index++;
            }
         },
         std::make_index_sequence<N>{});

      return pattern;
   }

   template <class TVec, class MaskGen>
   static TVec expand(const TVec& x, MaskGen g) {
      constexpr auto pattern = make_expand_mask<TVec::size>(g);

      using Mask = typename TVec::mask_type;
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16 || s == simd_shape::uint8x16) {
         TVec perm{[&pattern](auto i) { return pattern[i]; }};
         return vqtbl1q_s8(x, perm);
      }
   }

   template <class TVec, class I>
   static void partial_load(TVec& x, I first, std::iter_difference_t<I> n) {
      // we are not sure the be able to read the trailing bits without page fault.
      // so we will do an underaligned load and shift everything.

      // using T = typename TVec::value_type;
      // using U = details::unsigned_lane_t<T>;
      // using UVec = simd::rebind_t<U, TVec>;
      // using UMask = typename UVec::mask_type;

      // std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(std::to_address(first));

      // constexpr std::size_t page_size = 0x1000;  // 4096
      // constexpr std::uintptr_t page_mask = page_size - 1;
      // std::uintptr_t current_page = addr & ~page_mask;
      // std::uintptr_t next_page = current_page + page_size;
      // std::uintptr_t next_fault = next_page - addr;
      // // page fault if next_page is between addr + n and addr + TVec::size
      // bool page_fault = next_fault - n < TVec::size - n;
      // if (page_fault) {
      //    // In case of page fault, we are slow.
      //    // The read by itself is okayish, but the shift right will be awfully slow.
      //    std::uintptr_t overalignement = addr & (TVec::size - 1);
      //    auto aligned_ptr = std::to_address(first) - overalignement;

      //    UVec ramp_up{[](auto i) { return i; }};
      //    UMask aligned_mask = ramp_up < (n + overalignement);
      //    x.copy_from(aligned_ptr, aligned_mask);
      //    x = shift_right_lane(x, overalignement);
      // }
      // else {
      //    UVec ramp_up{[](auto i) { return i; }};
      //    UMask mask = ramp_up < n;
      //    x.copy_from(first, mask);
      // }
      x = TVec{0};
      std::memcpy(&x, std::to_address(first), n * sizeof(typename TVec::value_type));
   }
};


inline constexpr int arm_neon_register_size = 128 / 8;

template <class T>
struct select_neon_vector_type;

template <>
struct select_neon_vector_type<int8_t> {
   using type = int8x16_t;
};

template <>
struct select_neon_vector_type<int16_t> {
   using type = int16x8_t;
};

template <>
struct select_neon_vector_type<int32_t> {
   using type = int32x4_t;
};

template <>
struct select_neon_vector_type<int64_t> {
   using type = int64x2_t;
};

template <>
struct select_neon_vector_type<uint8_t> {
   using type = uint8x16_t;
};

template <>
struct select_neon_vector_type<uint16_t> {
   using type = uint16x8_t;
};

template <>
struct select_neon_vector_type<uint32_t> {
   using type = uint32x4_t;
};

template <>
struct select_neon_vector_type<uint64_t> {
   using type = uint64x2_t;
};

template <>
struct select_neon_vector_type<float> {
   using type = float32x4_t;
};

template <>
struct select_neon_vector_type<double> {
   using type = float64x2_t;
};


template <class T, int N = arm_neon_register_size / sizeof(T)>
struct neon_abi : public arm_neon_isa {
   using value_type = T;
   using value_storage_type = typename select_neon_vector_type<T>::type;


   using mask_lane_type = unsigned_lane_t<T>;
   using mask_storage_type = typename select_neon_vector_type<mask_lane_type>::type;

   static constexpr int size = N;
   static constexpr bool is_full = size == arm_neon_register_size / sizeof(T);
   static constexpr int width = 128;

   using isa_tag = neon_isa_tag;
};


}  // namespace details

}  // namespace simd

#endif
