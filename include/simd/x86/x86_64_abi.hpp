#ifndef SIMD_X86_X86_64_ABI_HPP
#define SIMD_X86_X86_64_ABI_HPP


#include <simd/details/bit.hpp>
#include <simd/details/simd_fwd.hpp>
#include <simd/details/simd_shape.hpp>
#include <simd/details/static_for_each.hpp>
#include <simd/x86/arch.hpp>

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>


namespace simd::details {



/// Multiplies signed 8-bit integer elements of two 256-bit vectors of
///    [32 x i8], and returns the lower 8 bits of each 16-bit product in the
///    [32 x i8] result.
///
/// \param a
///    A 256-bit vector of [32 x i8] containing one of the source operands.
/// \param b
///    A 256-bit vector of [32 x i8] containing one of the source operands.
/// \returns A 256-bit vector of [32 x i8] containing the products.
__m256i e_mm256_mullo_epi8(__m256i a, __m256i b) {
   __m256i a_lo = _mm256_cvtepi8_epi16(_mm256_extractf128_si256(a, 0));
   __m256i b_lo = _mm256_cvtepi8_epi16(_mm256_extractf128_si256(b, 0));

   __m256i lo = _mm256_and_si256(_mm256_mullo_epi16(a_lo, b_lo), _mm256_set1_epi16(0xFF));

   __m256i a_hi = _mm256_cvtepi8_epi16(_mm256_extractf128_si256(a, 1));
   __m256i b_hi = _mm256_cvtepi8_epi16(_mm256_extractf128_si256(b, 1));

   __m256i hi = _mm256_and_si256(_mm256_mullo_epi16(a_hi, b_hi), _mm256_set1_epi16(0xFF));

   return _mm256_or_si256(_mm256_slli_epi16(hi, 8), lo);
}

__m256i e_mm256_mullo_epi64(__m256i a, __m256i b) {
   // swap H<->L
   __m256i b_swap = _mm256_shuffle_epi32(b, _MM_SHUFFLE(2, 3, 0, 1));
   // 32-bit L*H and H*L cross-products
   __m256i crossprod = _mm256_mullo_epi32(a, b_swap);

   // bring the low half up to the top of each 64-bit chunk
   __m256i prodlh = _mm256_slli_epi64(crossprod, 32);
   // isolate the other, also into the high half were it needs to eventually be
   __m256i prodhl = _mm256_and_si256(crossprod, _mm256_set1_epi64x(0xFFFFFFFF00000000));
   // the sum of the cross products, with the low half of each u64 being 0.
   __m256i sumcross = _mm256_add_epi32(prodlh, prodhl);

   // widening 32x32 => 64-bit  low x low products
   __m256i prodll = _mm256_mul_epu32(a, b);
   // add the cross products into the high half of the result
   __m256i prod = _mm256_add_epi32(prodll, sumcross);
   return prod;
}


__m128i e_mm_setr_epi64x(long long x, long long y) {
   return _mm_set_epi64x(y, x);
}



struct x86_simd_isa {
   template <class TVec>
   static TVec make(typename TVec::value_type value) {
      constexpr simd_shape s = shape_of<TVec>();

      // SSE4.2
      if constexpr (s == simd_shape::int8x16)
         return _mm_set1_epi8(value);
      else if constexpr (s == simd_shape::int16x8)
         return _mm_set1_epi16(value);
      else if constexpr (s == simd_shape::int32x4)
         return _mm_set1_epi32(value);
      else if constexpr (s == simd_shape::int64x2)
         return _mm_set1_epi64x(value);
      else if constexpr (s == simd_shape::uint8x16)
         return _mm_set1_epi8(value);
      else if constexpr (s == simd_shape::uint16x8)
         return _mm_set1_epi16(value);
      else if constexpr (s == simd_shape::uint32x4)
         return _mm_set1_epi32(value);
      else if constexpr (s == simd_shape::uint64x2)
         return _mm_set1_epi64x(value);
      else if constexpr (s == simd_shape::float32x4)
         return _mm_set1_ps(value);
      else if constexpr (s == simd_shape::float64x2)
         return _mm_set1_pd(value);
      // AVX2
      else if constexpr (s == simd_shape::int8x32)
         return _mm256_set1_epi8(value);
      else if constexpr (s == simd_shape::int16x16)
         return _mm256_set1_epi16(value);
      else if constexpr (s == simd_shape::int32x8)
         return _mm256_set1_epi32(value);
      else if constexpr (s == simd_shape::int64x4)
         return _mm256_set1_epi64x(value);
      else if constexpr (s == simd_shape::uint8x32)
         return _mm256_set1_epi8(value);
      else if constexpr (s == simd_shape::uint16x16)
         return _mm256_set1_epi16(value);
      else if constexpr (s == simd_shape::uint32x8)
         return _mm256_set1_epi32(value);
      else if constexpr (s == simd_shape::uint64x4)
         return _mm256_set1_epi64x(value);
      else if constexpr (s == simd_shape::float32x8)
         return _mm256_set1_ps(value);
      else if constexpr (s == simd_shape::float64x4)
         return _mm256_set1_pd(value);
   }

   template <class TVec, class... Ts>
   static TVec make_vs(Ts... values) {
      constexpr simd_shape s = shape_of<TVec>();

      // SSE4.2
      if constexpr (s == simd_shape::int8x16)
         return _mm_setr_epi8(values...);
      else if constexpr (s == simd_shape::int16x8)
         return _mm_setr_epi16(values...);
      else if constexpr (s == simd_shape::int32x4)
         return _mm_setr_epi32(values...);
      else if constexpr (s == simd_shape::int64x2)
         return e_mm_setr_epi64x(values...);
      else if constexpr (s == simd_shape::uint8x16)
         return _mm_setr_epi8(values...);
      else if constexpr (s == simd_shape::uint16x8)
         return _mm_setr_epi16(values...);
      else if constexpr (s == simd_shape::uint32x4)
         return _mm_setr_epi32(values...);
      else if constexpr (s == simd_shape::uint64x2)
         return e_mm_setr_epi64x(values...);
      else if constexpr (s == simd_shape::float32x4)
         return _mm_setr_ps(values...);
      else if constexpr (s == simd_shape::float64x2)
         return _mm_setr_pd(values...);
      // AVX2
      else if constexpr (s == simd_shape::int8x32)
         return _mm256_setr_epi8(values...);
      else if constexpr (s == simd_shape::int16x16)
         return _mm256_setr_epi16(values...);
      else if constexpr (s == simd_shape::int32x8)
         return _mm256_setr_epi32(values...);
      else if constexpr (s == simd_shape::int64x4)
         return _mm256_setr_epi64x(values...);
      else if constexpr (s == simd_shape::uint8x32)
         return _mm256_setr_epi8(values...);
      else if constexpr (s == simd_shape::uint16x16)
         return _mm256_setr_epi16(values...);
      else if constexpr (s == simd_shape::uint32x8)
         return _mm256_setr_epi32(values...);
      else if constexpr (s == simd_shape::uint64x4)
         return _mm256_setr_epi64x(values...);
      else if constexpr (s == simd_shape::float32x8)
         return _mm256_setr_ps(values...);
      else if constexpr (s == simd_shape::float64x4)
         return _mm256_setr_pd(values...);
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
      constexpr int width = full_width<TVec>();

      auto fallback_load = [](auto it, const auto& m) {
         // If the ptr is aligned, we may be able to do a full read and apply a mask as
         // below.
         //
         // no direct intrincis, emulate it in 2ops. auto v =
         // _mm256_loadu_si256((__m256i*)std::to_address(it));
         // if constexpr (s == simd_shape::int16x16) {
         //    return _mm256_blend_epi16(v, _mm256_set1_epi16(0), m);
         // }
         // else if constexpr (s == simd_shape::int8x32) {
         //    return _mm256_blend_epi8(v, _mm256_set1_epi8(0), m);
         // }
         // a slow but safe emulation
         // typename TVec::value_type buffer[TVec::size];
         // for (std::size_t i = 0; i < TVec::size; ++i, ++it) {
         //    buffer[i] = m[i] ? *it : 0;
         // }

         // if constexpr (s == simd_shape::float32x4)
         //    return _mm_loadu_ps((const float*)&buffer);
         // else if constexpr (s == simd_shape::float64x2)
         //    return _mm_loadu_pd((const double*)&buffer);
         // if constexpr (width == 128)
         //    return _mm_loadu_si128((__m128i*)&buffer);
         // else if constexpr (width == 256)
         //    return _mm256_loadu_si256((__m256i*)&buffer);

         if constexpr (s == simd_shape::float32x4)
            return _mm_blendv_ps(
               _mm_loadu_ps(_mm_set1_ps(0), (const float*)std::to_address(it), m));
         else if constexpr (s == simd_shape::float64x2)
            return _mm_blendv_pd(_mm_set1_pd(0),
                                 _mm_loadu_pd((const double*)std::to_address(it)), m);
         if constexpr (width == 128)
            return _mm_blendv_epi8(_mm_set1_epi8(0),
                                   _mm_loadu_si128((__m128i*)std::to_address(it)), m);
         else if constexpr (width == 256)
            return _mm256_blendv_epi8(
               _mm256_set1_epi8(0), _mm256_loadu_si256((__m256i*)std::to_address(it)), m);
      };

      // notes: maskload require &*it to be aligned!
      if constexpr (current_isa >= x86_isa_level::avx2) {
         if constexpr (s == simd_shape::float32x4)
            return _mm_maskload_ps((const float*)std::to_address(it), m);  // avx2
         else if constexpr (s == simd_shape::float64x2)
            return _mm_maskload_pd((const double*)std::to_address(it), m);  // avx2
         else if constexpr (s == simd_shape::float32x8)
            return _mm256_maskload_ps((const float*)std::to_address(it), m);
         else if constexpr (s == simd_shape::float64x4)
            return _mm256_maskload_pd((const double*)std::to_address(it), m);
         else if constexpr (s == simd_shape::int64x2 || s == simd_shape::uint64x2)
            return _mm_maskload_epi64((std::int64_t const*)std::to_address(it), m);
         else if constexpr (s == simd_shape::int32x4 || s == simd_shape::uint32x4)
            return _mm_maskload_epi32((std::int32_t const*)std::to_address(it), m);
         else if constexpr (s == simd_shape::int64x4 || s == simd_shape::uint64x4)
            return _mm256_maskload_epi64((std::int64_t const*)std::to_address(it), m);
         else if constexpr (s == simd_shape::int32x8 || s == simd_shape::uint32x8)
            return _mm256_maskload_epi32((std::int32_t const*)std::to_address(it), m);
         else
            return fallback_load(it, m);
      }
      else
         return fallback_load(it, m);
   }

   template <class TVec, class It>
   static auto load(It it) {
      constexpr simd_shape s = shape_of<TVec>();
      constexpr int width = full_width<TVec>();

      if constexpr (is_full_vec<TVec>()) {
         if constexpr (s == simd_shape::float32x4)
            return _mm_loadu_ps(std::to_address(it));
         else if constexpr (s == simd_shape::float64x2)
            return _mm_loadu_pd(std::to_address(it));
         else if constexpr (s == simd_shape::float32x8)
            return _mm256_loadu_ps(std::to_address(it));
         else if constexpr (s == simd_shape::float64x4)
            return _mm256_loadu_pd(std::to_address(it));
         else if constexpr (width == 128)
            return _mm_loadu_si128((__m128i*)std::to_address(it));
         else if constexpr (width == 256)
            return _mm256_loadu_si256((__m256i*)std::to_address(it));
      }
      else {
         return load(it, [](auto i) { return i < TVec::size; });
      }
   }



   template <class TVec, class TMask, class Out>
   static void store(Out out, const TVec& self, const TMask& m) {
      constexpr simd_shape s = shape_of<TVec>();

      auto fallback_store = [](auto out, const auto& self, const auto& m) {
         // using value_type = typename TVec::value_tye;
         // std::array<value_type, TVec::size> buffer;
         // self.store(buffer.data());

         // using mask_value_type = typename TMask::mask_lane_type;
         // std::array<mask_value_type, TMask::size> mask;
         // std::memcpy(mask.data(), &m, sizeof(mask));

         // for (std::size_t i = 0; i < TVec::size; ++i, ++out) {
         //    if (mask[i])
         //       *out = buffer[i];
         // }
      };

      if constexpr (current_isa >= x86_isa_level::avx2) {
         if constexpr (s == simd_shape::float32x4)
            return _mm_maskstore_ps(std::to_address(out), m, self);
         else if constexpr (s == simd_shape::float64x2)
            return _mm_maskstore_pd(std::to_address(out), m, self);
         else if constexpr (s == simd_shape::float32x8)
            return _mm256_maskstore_ps(std::to_address(out), m, self);
         else if constexpr (s == simd_shape::float64x4)
            return _mm256_maskstore_pd(std::to_address(out), m, self);
         else if constexpr (s == simd_shape::int32x4 || s == simd_shape::uint32x4)
            return _mm_maskstore_epi32(std::to_address(out), m, self);
         else if constexpr (s == simd_shape::int64x2 || s == simd_shape::uint64x2)
            return _mm_maskstore_epi64(std::to_address(out), m, self);
         else if constexpr (s == simd_shape::int32x8 || s == simd_shape::uint32x8)
            return _mm256_maskstore_epi32(std::to_address(out), m, self);
         else if constexpr (s == simd_shape::int64x4 || s == simd_shape::uint64x4)
            return _mm256_maskstore_epi64(std::to_address(out), m, self);
         else
            return fallback_store(out, self, m);
      }
      else {
         return fallback_store(out, self, m);
      }
   }

   template <class TVec, class Out>
   static void store(Out out, const TVec& self) {
      constexpr simd_shape s = shape_of<TVec>();
      constexpr int width = full_width<TVec>();

      if constexpr (is_full_vec<TVec>()) {
         if constexpr (s == simd_shape::float32x4)
            return _mm_storeu_ps(std::to_address(out), self);
         else if constexpr (s == simd_shape::float64x2)
            return _mm_storeu_pd(std::to_address(out), self);
         if constexpr (s == simd_shape::float32x8)
            return _mm256_storeu_ps(std::to_address(out), self);
         else if constexpr (s == simd_shape::float64x4)
            return _mm256_storeu_pd(std::to_address(out), self);
         else if constexpr (width == 128)
            return _mm_storeu_si128((__m128i*)std::to_address(out), self);
         else if constexpr (width == 256)
            return _mm256_storeu_si256((__m256i*)std::to_address(out), self);
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

      // SSE4.2
      if constexpr (s == simd_shape::int8x16)
         return _mm_add_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return _mm_add_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return _mm_add_epi32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return _mm_add_epi64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return _mm_add_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return _mm_add_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return _mm_add_epi32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return _mm_add_epi64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4)
         return _mm_add_ps(lhs, rhs);
      else if constexpr (s == simd_shape::float64x2)
         return _mm_add_pd(lhs, rhs);
      // AVX2
      else if constexpr (s == simd_shape::int8x32)
         return _mm256_add_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x16)
         return _mm256_add_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x8)
         return _mm256_add_epi32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x4)
         return _mm256_add_epi64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x32)
         return _mm256_add_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x16)
         return _mm256_add_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x8)
         return _mm256_add_epi32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x4)
         return _mm256_add_epi64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x8)
         return _mm256_add_ps(lhs, rhs);
      else if constexpr (s == simd_shape::float64x4)
         return _mm256_add_pd(lhs, rhs);
   }

   template <class TVec>
   static TVec sub(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      // SSE4.2
      if constexpr (s == simd_shape::int8x16)
         return _mm_sub_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return _mm_sub_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return _mm_sub_epi32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2)
         return _mm_sub_epi64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return _mm_sub_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return _mm_sub_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return _mm_sub_epi32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x2)
         return _mm_sub_epi64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4)
         return _mm_sub_ps(lhs, rhs);
      else if constexpr (s == simd_shape::float64x2)
         return _mm_sub_pd(lhs, rhs);
      // AVX2
      else if constexpr (s == simd_shape::int8x32)
         return _mm256_sub_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x16)
         return _mm256_sub_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x8)
         return _mm256_sub_epi32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x4)
         return _mm256_sub_epi64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x32)
         return _mm256_sub_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x16)
         return _mm256_sub_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x8)
         return _mm256_sub_epi32(lhs, rhs);
      else if constexpr (s == simd_shape::uint64x4)
         return _mm256_sub_epi64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x8)
         return _mm256_sub_ps(lhs, rhs);
      else if constexpr (s == simd_shape::float64x4)
         return _mm256_sub_pd(lhs, rhs);
      else
         static_assert(std::is_same_v<void, TVec>, "Unsupported shape");
   }

   template <class TVec>
   static TVec mul(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      // SSE4.2
      if constexpr (s == simd_shape::int8x16 || s == simd_shape::uint8x16) {
         // return e_mm256_mullo_epi8(lhs, rhs); // ok on uint8??
         return apply_op(lhs, rhs, [](auto a, auto b) { return a * b; });
      }
      else if constexpr (s == simd_shape::int16x8 || s == simd_shape::uint16x8) {
         return _mm_mullo_epi16(lhs, rhs);
      }
      else if constexpr (s == simd_shape::int32x4 || s == simd_shape::uint32x4) {
         return _mm_mullo_epi32(lhs, rhs);
      }
      else if constexpr (s == simd_shape::int64x2 || s == simd_shape::uint64x2) {
         // return e_mm256_mullo_epi64(lhs, rhs); // ok on uint64
         return apply_op(lhs, rhs, [](auto a, auto b) { return a * b; });
      }
      else if constexpr (s == simd_shape::float32x4) {
         return _mm_mul_ps(lhs, rhs);
      }
      else if constexpr (s == simd_shape::float64x2) {
         return _mm_mul_pd(lhs, rhs);
      }
      // AVX2
      else if constexpr (s == simd_shape::int8x32 || s == simd_shape::uint8x32) {
         return e_mm256_mullo_epi8(lhs, rhs);  // ok on uint8??
      }
      else if constexpr (s == simd_shape::int16x16 || s == simd_shape::uint16x16) {
         return _mm256_mullo_epi16(lhs, rhs);
      }
      else if constexpr (s == simd_shape::int32x8 || s == simd_shape::uint32x8) {
         return _mm256_mullo_epi32(lhs, rhs);
      }
      else if constexpr (s == simd_shape::int64x4 || s == simd_shape::uint64x4) {
         return e_mm256_mullo_epi64(lhs, rhs);  // ok on uint64
      }
      else if constexpr (s == simd_shape::float32x8) {
         return _mm256_mul_ps(lhs, rhs);
      }
      else if constexpr (s == simd_shape::float64x4) {
         return _mm256_mul_pd(lhs, rhs);
      }
   }

   template <class TVec>
   static TVec div(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::float32x4) {
         return _mm_div_ps(lhs, rhs);
      }
      else if constexpr (s == simd_shape::float64x2) {
         return _mm_div_pd(lhs, rhs);
      }
      else if constexpr (s == simd_shape::float32x8) {
         return _mm256_div_ps(lhs, rhs);
      }
      else if constexpr (s == simd_shape::float64x4) {
         return _mm256_div_pd(lhs, rhs);
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
         return _mm_min_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return _mm_min_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return _mm_min_epi32(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return _mm_min_epu8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return _mm_min_epu16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return _mm_min_epu32(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4)
         return _mm_min_ps(lhs, rhs);
      else if constexpr (s == simd_shape::float64x2)
         return _mm_min_pd(lhs, rhs);
      else if constexpr (s == simd_shape::int8x32)
         return _mm256_min_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x16)
         return _mm256_min_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x8)
         return _mm256_min_epi32(lhs, rhs);
      else if constexpr (current_isa >= x86_isa_level::avx512 && s == simd_shape::int64x4)
         return _mm256_min_epi64(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x32)
         return _mm256_min_epu8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x16)
         return _mm256_min_epu16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x8)
         return _mm256_min_epu32(lhs, rhs);
      else if constexpr (current_isa >= x86_isa_level::avx512 &&
                         s == simd_shape::uint64x4)
         return _mm256_min_epu64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x8)
         return _mm256_min_ps(lhs, rhs);
      else if constexpr (s == simd_shape::float64x4)
         return _mm256_min_pd(lhs, rhs);
      // else
      //    return simd::select(lhs < rhs, lhs, rhs);
   }

   template <class TVec>
   static TVec max(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16)
         return _mm_max_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8)
         return _mm_max_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4)
         return _mm_max_epi32(lhs, rhs);
      else if constexpr (s == simd_shape::uint8x16)
         return _mm_max_epu8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x8)
         return _mm_max_epu16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x4)
         return _mm_max_epu32(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4)
         return _mm_max_ps(lhs, rhs);
      else if constexpr (s == simd_shape::float64x2)
         return _mm_max_pd(lhs, rhs);
      else if constexpr (s == simd_shape::int8x32)
         return _mm256_max_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x16)
         return _mm256_max_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x8)
         return _mm256_max_epi32(lhs, rhs);
      else if constexpr (current_isa >= x86_isa_level::avx512 && s == simd_shape::int64x4)
         return _mm256_max_epi64(lhs, rhs);  // avx512
      else if constexpr (s == simd_shape::uint8x32)
         return _mm256_max_epu8(lhs, rhs);
      else if constexpr (s == simd_shape::uint16x16)
         return _mm256_max_epu16(lhs, rhs);
      else if constexpr (s == simd_shape::uint32x8)
         return _mm256_max_epu32(lhs, rhs);
      else if constexpr (current_isa >= x86_isa_level::avx512 &&
                         s == simd_shape::uint64x4)
         return _mm256_max_epu64(lhs, rhs);  // avx512
      else if constexpr (s == simd_shape::float32x8)
         return _mm256_max_ps(lhs, rhs);
      else if constexpr (s == simd_shape::float64x4)
         return _mm256_max_pd(lhs, rhs);
      // else
      //    return simd::select(lhs > rhs, lhs, rhs);
   }

   template <class TVec>
   static TVec unary_minus(const TVec& self) {
      return TVec{0} - self;
   }

   template <class TVec>
   static TVec bit_not(const TVec& self) {
      // return simd_int8{0xFF} ^ *this;
      using value_type = typename TVec::value_type;
      constexpr auto all_bits_set = details::set_allbits<value_type>();
      return bit_xor(TVec{static_cast<value_type>(all_bits_set)}, self);
   }

   template <class TVec>
   static TVec bit_and(const TVec& lhs, const TVec& rhs) {
      constexpr int width = full_width<TVec>();

      if constexpr (width == 128)
         return _mm_and_si128(lhs, rhs);
      else if constexpr (width == 256)
         return _mm256_and_si256(lhs, rhs);
   }

   template <class TVec>
   static TVec bit_or(const TVec& lhs, const TVec& rhs) {
      constexpr int width = full_width<TVec>();

      if constexpr (width == 128)
         return _mm_or_si128(lhs, rhs);
      else if constexpr (width == 256)
         return _mm256_or_si256(lhs, rhs);
   }

   template <class TVec>
   static TVec bit_xor(const TVec& lhs, const TVec& rhs) {
      constexpr int width = full_width<TVec>();

      if constexpr (width == 128)
         return _mm_xor_si128(lhs, rhs);
      else if constexpr (width == 256)
         return _mm256_xor_si256(lhs, rhs);
   }

   template <class TVec>
   static TVec bit_shift_left(const TVec& lhs, int count) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16 || s == simd_shape::uint8x16) {
         if (count > 8)
            return 0;
         __m128i m0 = _mm_set1_epi16(0x00FF);
         __m128i m1 = _mm_set1_epi16(0xFF00);
         auto p0 = _mm_and_si128(_mm_slli_epi16(_mm_and_si128(lhs, m0), count), m0);
         auto p1 = _mm_and_si128(_mm_slli_epi16(_mm_and_si128(lhs, m1), count), m1);
         return _mm_or_si128(p0, p1);
      }
      else if constexpr (s == simd_shape::int16x8 || s == simd_shape::uint16x8) {
         return _mm_slli_epi16(lhs, count);
      }
      else if constexpr (s == simd_shape::int32x4 || s == simd_shape::uint32x4) {
         return _mm_slli_epi32(lhs, count);
      }
      else if constexpr (s == simd_shape::int64x2 || s == simd_shape::uint64x2) {
         return _mm_slli_epi64(lhs, count);
      }
      else if constexpr (s == simd_shape::int8x32 || s == simd_shape::uint8x32) {
         if (count > 8)
            return 0;
         __m256i m0 = _mm256_set1_epi16(0x00FF);
         __m256i m1 = _mm256_set1_epi16(0xFF00);
         auto p0 =
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(lhs, m0), count), m0);
         auto p1 =
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(lhs, m1), count), m1);
         return _mm256_or_si256(p0, p1);
      }
      else if constexpr (s == simd_shape::int16x16 || s == simd_shape::uint16x16) {
         return _mm256_slli_epi16(lhs, count);
      }
      else if constexpr (s == simd_shape::int32x8 || s == simd_shape::uint32x8) {
         return _mm256_slli_epi32(lhs, count);
      }
      else if constexpr (s == simd_shape::int64x4 || s == simd_shape::uint64x4) {
         return _mm256_slli_epi64(lhs, count);
      }
   }

   template <class TVec>
   static TVec bit_shift_right(const TVec& lhs, int count) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16 || s == simd_shape::uint8x16) {
         if (count > 8)
            return 0;
         __m128i m0 = _mm_set1_epi16(0x00FF);
         __m128i m1 = _mm_set1_epi16(0xFF00);
         auto p0 = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(lhs, m0), count), m0);
         auto p1 = _mm_and_si128(_mm_srli_epi16(_mm_and_si128(lhs, m1), count), m1);
         return _mm_or_si128(p0, p1);
      }
      else if constexpr (s == simd_shape::int16x8 || s == simd_shape::uint16x8) {
         return _mm_srli_epi16(lhs, count);
      }
      else if constexpr (s == simd_shape::int32x4 || s == simd_shape::uint32x4) {
         return _mm_srli_epi32(lhs, count);
      }
      else if constexpr (s == simd_shape::int64x2 || s == simd_shape::uint64x2) {
         return _mm_srli_epi64(lhs, count);
      }
      else if constexpr (s == simd_shape::int8x32 || s == simd_shape::uint8x32) {
         if (count > 8)
            return 0;
         __m256i m0 = _mm256_set1_epi16(0x00FF);
         __m256i m1 = _mm256_set1_epi16(0xFF00);
         auto p0 =
            _mm256_and_si256(_mm256_srli_epi16(_mm256_and_si256(lhs, m0), count), m0);
         auto p1 =
            _mm256_and_si256(_mm256_srli_epi16(_mm256_and_si256(lhs, m1), count), m1);
         return _mm256_or_si256(p0, p1);
      }
      else if constexpr (s == simd_shape::int16x16 || s == simd_shape::uint16x16) {
         return _mm256_srli_epi16(lhs, count);
      }
      else if constexpr (s == simd_shape::int32x8 || s == simd_shape::uint32x8) {
         return _mm256_srli_epi32(lhs, count);
      }
      else if constexpr (s == simd_shape::int64x4 || s == simd_shape::uint64x4) {
         return _mm256_srli_epi64(lhs, count);
      }
   }

   template <class TVec>
   static auto cmp_eq(const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (s == simd_shape::int8x16 || s == simd_shape::uint8x16)
         return _mm_cmpeq_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x8 || s == simd_shape::uint16x8)
         return _mm_cmpeq_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x4 || s == simd_shape::uint32x4)
         return _mm_cmpeq_epi32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x2 || s == simd_shape::uint64x2)
         return _mm_cmpeq_epi64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x4)
         return reinterpret_cast<__m128i>(_mm_cmpeq_ps(lhs, rhs));
      else if constexpr (s == simd_shape::float64x2)
         return reinterpret_cast<__m128i>(_mm_cmpeq_pd(lhs, rhs));
      else if constexpr (s == simd_shape::int8x32 || s == simd_shape::uint8x32)
         return _mm256_cmpeq_epi8(lhs, rhs);
      else if constexpr (s == simd_shape::int16x16 || s == simd_shape::uint16x16)
         return _mm256_cmpeq_epi16(lhs, rhs);
      else if constexpr (s == simd_shape::int32x8 || s == simd_shape::uint32x8)
         return _mm256_cmpeq_epi32(lhs, rhs);
      else if constexpr (s == simd_shape::int64x4 || s == simd_shape::uint64x4)
         return _mm256_cmpeq_epi64(lhs, rhs);
      else if constexpr (s == simd_shape::float32x8)
         return reinterpret_cast<__m256i>(_mm256_cmp_ps(lhs, rhs, 0x00));
      else if constexpr (s == simd_shape::float64x4)
         return reinterpret_cast<__m256i>(_mm256_cmp_pd(lhs, rhs, 0x00));
   }

   template <class TVec>
   static auto cmp_lt(const TVec& lhs, const TVec& rhs) {
      using T = typename TVec::value_type;
      using TVecMask = typename TVec::mask_type;
      constexpr simd_shape s = shape_of<TVec>();

      if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
         using SignedT = std::make_signed_t<T>;
         using SignedTVec = rebind_t<SignedT, TVec>;

         constexpr SignedT offset =
            static_cast<SignedT>(static_cast<SignedT>(1) << (8 * sizeof(SignedT) - 1));
         SignedTVec a = std::bit_cast<SignedTVec>(lhs) - offset;
         SignedTVec b = std::bit_cast<SignedTVec>(rhs) - offset;
         return std::bit_cast<TVecMask>(a < b);
      }
      else if constexpr (s == simd_shape::int8x16)
         return _mm_cmpgt_epi8(rhs, lhs);
      else if constexpr (s == simd_shape::int16x8)
         return _mm_cmpgt_epi16(rhs, lhs);
      else if constexpr (s == simd_shape::int32x4)
         return _mm_cmpgt_epi32(rhs, lhs);
      else if constexpr (s == simd_shape::int64x2)
         return _mm_cmpgt_epi64(rhs, lhs);
      // else if constexpr (s == simd_shape::uint8x16)
      //    return e_mm_cmpgt_epu8(rhs, lhs);
      // else if constexpr (s == simd_shape::uint16x8)
      //    return e_mm_cmpgt_epu16(rhs, lhs);
      // else if constexpr (s == simd_shape::uint32x4)
      //    return e_mm_cmpgt_epu32(rhs, lhs);
      // else if constexpr (s == simd_shape::uint64x2) {
      //    auto offset = _mm_set1_epi64x(0x8000'0000'0000'0000);
      //    return _mm_cmpgt_epi64(_mm_sub_epi64(rhs, offset), _mm_sub_epi64(lhs,
      //    offset));
      // }
      else if constexpr (s == simd_shape::float32x4)
         return reinterpret_cast<__m128i>(_mm_cmplt_ps(lhs, rhs));
      else if constexpr (s == simd_shape::float64x2)
         return reinterpret_cast<__m128i>(_mm_cmplt_pd(lhs, rhs));
      else if constexpr (s == simd_shape::int8x32)
         return _mm256_cmpgt_epi8(rhs, lhs);
      else if constexpr (s == simd_shape::int16x16)
         return _mm256_cmpgt_epi16(rhs, lhs);
      else if constexpr (s == simd_shape::int32x8)
         return _mm256_cmpgt_epi32(rhs, lhs);
      else if constexpr (s == simd_shape::int64x4)
         return _mm256_cmpgt_epi64(rhs, lhs);
      // else if constexpr (s == simd_shape::uint8x32)
      //    return e_mm256_cmpgt_epu8(rhs, lhs);
      // else if constexpr (s == simd_shape::uint16x16)
      //    return e_mm256_cmpgt_epu16(rhs, lhs);
      // else if constexpr (s == simd_shape::uint32x8)
      //    return e_mm256_cmpgt_epu32(rhs, lhs);
      // // else if constexpr (s == simd_shape::uint64x4)
      // //    return _mm256_cmpgt_epu64(rhs, lhs);
      // else if constexpr (s == simd_shape::uint64x4) {
      //    auto offset = _mm256_set1_epi64x(0x8000'0000'0000'0000);
      //    return _mm256_cmpgt_epi64(_mm256_sub_epi64(rhs, offset),
      //                              _mm256_sub_epi64(lhs, offset));
      // }
      else if constexpr (s == simd_shape::float32x8)
         return reinterpret_cast<__m256i>(_mm256_cmp_ps(lhs, rhs, 0x01));
      else if constexpr (s == simd_shape::float64x4)
         return reinterpret_cast<__m256i>(_mm256_cmp_pd(lhs, rhs, 0x01));
      // else {
      // (as_signed(lhs) - 0x80) < (as_signed(rhs) - 0x80)
      //    // unsigned comparison
      //    //
      //    // _mm_cmpgt_epu8 translated to avx2 + using simd api from
      //    // http://www.alfredklomp.com/programming/sse-intrinsics/
      //    //
      //    return _mm256_andnot_si256(rhs == lhs, max(rhs, lhs) == rhs);
      // }
   }

   template <class TVec, class TVecMask>
   static auto blend(const TVecMask& m, const TVec& lhs, const TVec& rhs) {
      constexpr simd_shape s = shape_of<TVec>();
      constexpr int width = full_width<TVec>();

      if constexpr (s == simd_shape::float32x4)
         return _mm_blendv_ps(lhs, rhs, m);
      else if constexpr (s == simd_shape::float64x2)
         return _mm_blendv_pd(lhs, rhs, m);
      else if constexpr (s == simd_shape::float32x8)
         return _mm256_blendv_ps(lhs, rhs, m);
      else if constexpr (s == simd_shape::float64x4)
         return _mm256_blendv_pd(lhs, rhs, m);
      else if constexpr (width == 128)
         // interpreting 16, 32 and 64bits mask as 8bits is ok
         return _mm_blendv_epi8(lhs, rhs, m);
      else if constexpr (width == 256)
         // interpreting 16, 32 and 64bits mask as 8bits is ok
         return _mm256_blendv_epi8(lhs, rhs, m);
   }

   template <class TVec>
   static auto get_lane(const TVec& self, std::size_t index) {
      // TODO assert
      // memcpy or store?
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
      constexpr int width = full_width<TMask>();

      if constexpr (width == 128)
         return _mm_set1_epi8(value ? details::set_allbits<std::int8_t>() : 0);
      else if constexpr (width == 256)
         return _mm256_set1_epi8(value ? details::set_allbits<std::int8_t>() : 0);
   }

   template <class TMask, class... Ts>
   static TMask make_mask_vs(Ts... values) {
      constexpr simd_shape s = shape_of<typename TMask::abi_type>();

      // SSE4.2
      if constexpr (s == simd_shape::int8x16)
         return _mm_setr_epi8(values...);
      else if constexpr (s == simd_shape::int16x8)
         return _mm_setr_epi16(values...);
      else if constexpr (s == simd_shape::int32x4)
         return _mm_setr_epi32(values...);
      else if constexpr (s == simd_shape::int64x2)
         return e_mm_setr_epi64x(values...);
      else if constexpr (s == simd_shape::uint8x16)
         return _mm_setr_epi8(values...);
      else if constexpr (s == simd_shape::uint16x8)
         return _mm_setr_epi16(values...);
      else if constexpr (s == simd_shape::uint32x4 || s == simd_shape::float32x4)
         return _mm_setr_epi32(values...);
      else if constexpr (s == simd_shape::uint64x2 || s == simd_shape::float64x2)
         return e_mm_setr_epi64x(values...);
      // else if constexpr (s == simd_shape::float32x4)
      //    return _mm_setr_ps(values...);
      // else if constexpr (s == simd_shape::float64x2)
      //    return _mm_setr_pd(values...);
      // AVX2
      else if constexpr (s == simd_shape::int8x32)
         return _mm256_setr_epi8(values...);
      else if constexpr (s == simd_shape::int16x16)
         return _mm256_setr_epi16(values...);
      else if constexpr (s == simd_shape::int32x8)
         return _mm256_setr_epi32(values...);
      else if constexpr (s == simd_shape::int64x4)
         return _mm256_setr_epi64x(values...);
      else if constexpr (s == simd_shape::uint8x32)
         return _mm256_setr_epi8(std::bit_cast<std::int8_t>(values)...);
      else if constexpr (s == simd_shape::uint16x16)
         return _mm256_setr_epi16(std::bit_cast<std::int16_t>(values)...);
      else if constexpr (s == simd_shape::uint32x8 || s == simd_shape::float32x8)
         return _mm256_setr_epi32(std::bit_cast<std::int32_t>(values)...);
      else if constexpr (s == simd_shape::uint64x4 || s == simd_shape::float64x4)
         return _mm256_setr_epi64x(std::bit_cast<std::int64_t>(values)...);
      // else if constexpr (s == simd_shape::float32x8)
      //    return _mm256_setr_ps(values...);
      // else if constexpr (s == simd_shape::float64x4)
      //    return _mm256_setr_pd(values...);
   }

   template <class TMask, class G, std::size_t... Is>
   static TMask make_gen_mask(G&& gen, std::index_sequence<Is...>) {
      using value_type_vec = typename TMask::abi_type::value_type;
      if constexpr (std::is_floating_point_v<value_type_vec>) {
         value_type_vec one = details::set_allbits<value_type_vec>();
         constexpr value_type_vec zero = 0;
         return make_mask_vs<TMask>(
            (static_cast<bool>(gen(std::integral_constant<int, Is>{})) ? one : zero)...);
      }
      else {
         constexpr value_type_vec one = details::set_allbits<value_type_vec>();
         constexpr value_type_vec zero = 0;
         return make_mask_vs<TMask>(
            (static_cast<bool>(gen(std::integral_constant<int, Is>{})) ? one : zero)...);
      }
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
      typename TMask::mask_lane_type data[TMask::size];
      for (int i = 0; i < TMask::size; ++i, ++it)
         data[i] = m[i] ? (*it ? details::set_allbits<std::int8_t>() : 0) : 0;

      constexpr int width = full_width<TMask>();
      if constexpr (width == 128)
         return _mm_loadu_si128((__m128i*)data);
      else if constexpr (width == 256)
         return _mm256_loadu_si256((__m256i*)data);
   }

   template <class TMask, class It>
   static TMask load_mask(It it) {
      if constexpr (is_full_vec<TMask>()) {
         typename TMask::mask_lane_type data[TMask::size];
         for (int i = 0; i < TMask::size; ++i, ++it)
            data[i] = *it ? details::set_allbits<std::int8_t>() : 0;

         constexpr int width = full_width<TMask>();
         if constexpr (width == 128)
            return _mm_loadu_si128((__m128i*)data);
         else if constexpr (width == 256)
            return _mm256_loadu_si256((__m256i*)data);
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
            *out = static_cast<bool>(data[i]);
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
      constexpr int width = full_width<TMask>();

      // compare bits by bits?
      if constexpr (width == 128)
         return _mm_cmpeq_epi8(lhs, rhs);
      else if constexpr (width == 256)
         return _mm256_cmpeq_epi8(lhs, rhs);
   }

   template <class TMask>
   static bool mask_get_lane(const TMask& self, std::size_t index) {
      // could deduce target_value_type from avx registry size and mask size:
      //    256 / TMask::size
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
      // return mask_bit_xor(TMask{true}, self);
      using abi = typename TMask::abi_type;
      using mask_storage_type = typename abi::mask_storage_type;
      const mask_storage_type& s = self;
      return ~s;
   }

   template <class TMask>
   static TMask mask_bit_and(const TMask& lhs, const TMask& rhs) {
      return bit_and(lhs, rhs);
   }

   template <class TMask>
   static TMask mask_bit_or(const TMask& lhs, const TMask& rhs) {
      return bit_or(lhs, rhs);
   }

   template <class TMask>
   static TMask mask_bit_xor(const TMask& lhs, const TMask& rhs) {
      return bit_xor(lhs, rhs);
   }

   template <class TMask>
   static bool all_of(const TMask& self) {
      constexpr int width = full_width<TMask>();

      if constexpr (width == 128)
         return _mm_testc_si128(self, make_mask<TMask>(true)) != 0;
      else if constexpr (width == 256)
         return _mm256_testc_si256(self, make_mask<TMask>(true)) != 0;
   }

   template <class TMask>
   static bool is_zero(const TMask& k) {
      constexpr int width = full_width<TMask>();

      if constexpr (width == 128)
         return _mm_testz_si128(k, k);
      else if constexpr (width == 256)
         return _mm256_testz_si256(k, k);
   }

   template <class TMask>
   static int reduce_mask(const TMask& k) {
      constexpr simd_shape s = shape_of<typename TMask::abi_type>();

      if constexpr (s == simd_shape::uint8x16 || s == simd_shape::int8x16) {
         return _mm_movemask_epi8(k);
      }
      else if constexpr (s == simd_shape::uint8x32 || s == simd_shape::int8x32) {
         return _mm256_movemask_epi8(k);
      }
      else if constexpr (s == simd_shape::uint16x8 || s == simd_shape::int16x8) {
         return _mm_movemask_epi8(_mm_packs_epi16(k, _mm_setzero_si128()));
      }
      else if constexpr (s == simd_shape::uint16x16 || s == simd_shape::int16x16) {
         auto lo = _mm256_extracti128_si256(k, 0);
         auto hi = _mm256_extracti128_si256(k, 1);
         auto m_lo = _mm_movemask_epi8(_mm_packs_epi16(lo, _mm_setzero_si128()));
         auto m_hi = _mm_movemask_epi8(_mm_packs_epi16(hi, _mm_setzero_si128()));
         return m_lo | (m_hi << 8);
      }
      else if constexpr (s == simd_shape::uint32x4 || s == simd_shape::int32x4 ||
                         s == simd_shape::float32x4) {
         return _mm_movemask_ps(reinterpret_cast<__m128>(static_cast<__m128i>(k)));
      }
      else if constexpr (s == simd_shape::uint32x8 || s == simd_shape::int32x8 ||
                         s == simd_shape::float32x8) {
         return _mm256_movemask_ps(reinterpret_cast<__m256>(static_cast<__m256i>(k)));
      }
      else if constexpr (s == simd_shape::uint64x2 || s == simd_shape::int64x2 ||
                         s == simd_shape::float64x2) {
         return _mm_movemask_pd(reinterpret_cast<__m128d>(static_cast<__m128i>(k)));
      }
      else if constexpr (s == simd_shape::uint64x4 || s == simd_shape::int64x4 ||
                         s == simd_shape::float64x4) {
         return _mm256_movemask_pd(reinterpret_cast<__m256d>(static_cast<__m256i>(k)));
      }
   }

   template <class TMask>
   static int reduce_count(const TMask& k) {
      int m = reduce_mask(k);
      return std::popcount(static_cast<unsigned int>(m));
   }

   template <class TMask>
   static int reduce_min_index(const TMask& k) {
      int m = reduce_mask(k);
      return lowest_bit_set(m);
   }

   template <class TMask>
   static int reduce_max_index(const TMask& k) {
      int m = reduce_mask(k);
      return highest_bit_set(m);
   }

   template <class TVec>
   static TVec byteswap(const TVec& x) {
      constexpr simd_shape s = shape_of<TVec>();
      if constexpr (s == simd_shape::int16x8 || s == simd_shape::uint16x8) {
         return _mm_shuffle_epi8(
            x, _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14));
      }
      else if constexpr (s == simd_shape::int32x4 || s == simd_shape::uint32x4) {
         return _mm_shuffle_epi8(
            x, _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12));
      }
      else if constexpr (s == simd_shape::int64x2 || s == simd_shape::uint64x2) {
         return _mm_shuffle_epi8(
            x, _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8));
      }
      else if constexpr (s == simd_shape::int16x16 || s == simd_shape::uint16x16) {
         return _mm256_shuffle_epi8(
            x,
            _mm256_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,  //
                             1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14));
      }
      else if constexpr (s == simd_shape::int32x8 || s == simd_shape::uint32x8) {
         return _mm256_shuffle_epi8(
            x,
            _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,  //
                             3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12));
      }
      else if constexpr (s == simd_shape::int64x4 || s == simd_shape::uint64x4) {
         return _mm256_shuffle_epi8(
            x,
            _mm256_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,  //
                             7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8));
      }
   }

   template <std::size_t N, class MaskGen>
   static constexpr auto make_compress_mask(MaskGen g) {
      std::array<int, N> pattern{-1};
      std::array<int, N> lo_perm{-1};
      std::array<bool, N> lo_mask{false};
      // std::array<int, N / 2> hi_perm{-1};
      // std::array<bool, N / 2> hi_mask{false};

      std::array<int, N> hi_to_lo_perm{-1};
      std::array<bool, N> hi_to_lo_mask{false};

      std::ranges::fill_n(pattern.begin(), N, -1);
      std::ranges::fill_n(lo_perm.begin(), N, -1);
      // std::ranges::fill_n(hi_perm.begin(), N / 2, -1);
      std::ranges::fill_n(hi_to_lo_perm.begin(), N, -1);
      std::ranges::fill_n(lo_mask.begin(), N, false);
      // std::ranges::fill_n(hi_mask.begin(), N / 2, false);
      std::ranges::fill_n(hi_to_lo_mask.begin(), N, false);

      int index = 0;
      for_template(
         [&](auto i) {
            bool in_lo = i <= 15 && index <= 15;
            bool in_hi = i > 15 && index > 15;
            // bool cross_lane = index <= 15 && i > 15;

            if (g(i)) {
               if (in_lo || in_hi) {
                  lo_perm[index] = i;
                  lo_mask[index] = true;
               }
               // else if (in_hi) {
               //    hi_perm[index % (N / 2)] = i;
               //    hi_mask[index % (N / 2)] = true;
               // }
               else /*if (cross_lane)*/ {
                  hi_to_lo_perm[index] = i % 16;
                  hi_to_lo_mask[index] = true;
               }
               pattern[index++] = i;
            }
         },
         std::make_index_sequence<N>{});

      return std::make_tuple(pattern, lo_perm, lo_mask, hi_to_lo_perm, hi_to_lo_mask);
   }

   template <class TVec, class MaskGen>
   static TVec compress(const TVec& x, MaskGen g) {
      using Abi = typename TVec::abi_type;
      if constexpr (has_isa_tag<Abi, avx2_isa_tag>) {
         constexpr auto pattern = make_compress_mask<TVec::size>(g);
         constexpr auto lo_perm_pattern = std::get<1>(pattern);
         constexpr auto lo_mask_pattern = std::get<2>(pattern);
         // constexpr auto hi_perm = std::get<3>(pattern);
         // constexpr auto hi_mask = std::get<4>(pattern);
         constexpr auto hi_to_lo_pattern = std::get<3>(pattern);
         // constexpr auto hi_to_lo_mask = std::get<4>(pattern);

         // constexpr std::size_t trunc_size = 24;  // std::ranges::count(lo_mask_pattern,
         // 0);


         // TODO: rebuild register based on perm + mask
         // call the right shuffle

         using Mask = typename TVec::mask_type;
         constexpr simd_shape s = shape_of<TVec>();
         // using CompressSimd = std::resize_simd_t<24, Simd>;

         if constexpr (s == simd_shape::int8x32 || s == simd_shape::uint8x32) {
            TVec lo_hi_perm{[&lo_perm_pattern](auto i) { return lo_perm_pattern[i]; }};

            // Shuffle on lo and hi lanes
            TVec lo_hi = _mm256_shuffle_epi8(x, lo_hi_perm);

            // Permute lanes
            // _mm256_permutexvar_epi8(int A, int B) ??
            TVec hi_none = _mm256_inserti128_si256(x, _mm256_extracti128_si256(x, 1), 0);
            TVec hi_to_lo_perm{
               [&hi_to_lo_pattern](auto i) { return hi_to_lo_pattern[i]; }};
            hi_none = _mm256_shuffle_epi8(hi_none, hi_to_lo_perm);

            Mask m{[&lo_mask_pattern](auto i) { return lo_mask_pattern[i]; }};
            return blend(m, hi_none, lo_hi);
         }
      }
      else {
         constexpr auto patterns = make_compress_mask<TVec::size>(g);
         constexpr auto pattern = std::get<0>(patterns);

         constexpr simd_shape s = shape_of<TVec>();

         if constexpr (s == simd_shape::int8x16 || s == simd_shape::uint8x16) {
            TVec perm{[&pattern](auto i) { return pattern[i]; }};
            return _mm_shuffle_epi8(x, perm);
         }
      }
   }


   template <std::size_t N, class MaskGen>
   static constexpr auto make_expand_mask(MaskGen g) {
      std::array<int, N> pattern{-1};
      std::array<int, N> lo_perm{-1};
      std::array<bool, N> lo_mask{false};
      // std::array<int, N / 2> hi_perm{-1};
      // std::array<bool, N / 2> hi_mask{false};

      std::array<int, N> lo_to_hi_perm{-1};
      std::array<bool, N> lo_to_hi_mask{false};

      std::ranges::fill_n(pattern.begin(), N, -1);
      std::ranges::fill_n(lo_perm.begin(), N, -1);
      // std::ranges::fill_n(hi_perm.begin(), N / 2, -1);
      std::ranges::fill_n(lo_to_hi_perm.begin(), N, -1);
      std::ranges::fill_n(lo_mask.begin(), N, false);
      // std::ranges::fill_n(hi_mask.begin(), N / 2, false);
      std::ranges::fill_n(lo_to_hi_mask.begin(), N, false);

      int index = 0;
      for_template(
         [&](auto i) {
            bool in_lo = i <= 15 && index <= 15;
            bool in_hi = i > 15 && index > 15;
            // bool cross_lane = index <= 15 && i > 15;

            if (g(i)) {
               if (in_lo || in_hi) {
                  lo_perm[i] = index;
                  lo_mask[i] = true;
               }
               // else if (in_hi) {
               //    hi_perm[index % (N / 2)] = i;
               //    hi_mask[index % (N / 2)] = true;
               // }
               else /*if (cross_lane)*/ {
                  lo_to_hi_perm[i] = index % 16;
                  lo_to_hi_mask[i] = true;
               }
               pattern[i] = index++;
            }
         },
         std::make_index_sequence<N>{});

      return std::make_tuple(pattern, lo_perm, lo_mask, lo_to_hi_perm, lo_to_hi_mask);
   }

   template <class TVec, class MaskGen>
   static TVec expand(const TVec& x, MaskGen g) {
      using Abi = typename TVec::abi_type;
      if constexpr (has_isa_tag<Abi, avx2_isa_tag>) {
         constexpr auto pattern = make_expand_mask<TVec::size>(g);
         constexpr auto lo_perm_pattern = std::get<1>(pattern);
         constexpr auto lo_mask_pattern = std::get<2>(pattern);
         // constexpr auto hi_perm = std::get<3>(pattern);
         // constexpr auto hi_mask = std::get<4>(pattern);
         constexpr auto lo_to_hi_pattern = std::get<3>(pattern);

         using Mask = typename TVec::mask_type;

         TVec lo_hi_perm{[&lo_perm_pattern](auto i) { return lo_perm_pattern[i]; }};

         // Shuffle on lo and hi lanes
         TVec lo_hi = _mm256_shuffle_epi8(x, lo_hi_perm);

         // Permute lanes
         // _mm256_permutexvar_epi8(int A, int B) ??
         TVec lo_none = _mm256_inserti128_si256(x, _mm256_extracti128_si256(x, 0), 1);
         TVec lo_to_hi_perm{[&lo_to_hi_pattern](auto i) { return lo_to_hi_pattern[i]; }};
         lo_none = _mm256_shuffle_epi8(lo_none, lo_to_hi_perm);

         Mask m{[&lo_mask_pattern](auto i) { return lo_mask_pattern[i]; }};
         auto r = blend(m, lo_none, lo_hi);
         return r;
      }
      else {
         constexpr auto patterns = make_expand_mask<TVec::size>(g);
         constexpr auto pattern = std::get<0>(patterns);

         constexpr simd_shape s = shape_of<TVec>();

         if constexpr (s == simd_shape::int8x16 || s == simd_shape::uint8x16) {
            TVec perm{[&pattern](auto i) { return pattern[i]; }};
            return _mm_shuffle_epi8(x, perm);
         }
      }
   }

   template <class TVec, class I>
   static void partial_load(TVec& x, I first, std::iter_difference_t<I> n) {
      // UVec ramp_up{[](auto i) { return i; }};
      // UMask mask = ramp_up < n;
      // x.copy_from(first, mask);
      x = TVec{0};
      std::memcpy(&x, std::to_address(first), n * sizeof(typename TVec::value_type));
   }
};



template <class T, int N = sizeof(__m128) / sizeof(T)>
struct sse4_abi : public x86_simd_isa {
   using value_type = T;
   using value_storage_type =
      std::conditional_t<std::is_same_v<double, T>, __m128d,
                         std::conditional_t<std::is_same_v<float, T>, __m128, __m128i>>;

   using mask_storage_type = __m128i;
   using mask_lane_type = unsigned_lane_t<T>;

   // lane number
   static constexpr int size = N;
   static constexpr bool is_full = size == sizeof(__m128) / sizeof(T);
   static constexpr int width = 128;

   using isa_tag = sse4_isa_tag;
};

template <class T, int N = sizeof(__m256) / sizeof(T)>
struct avx2_abi : public x86_simd_isa {
   using value_type = T;
   using value_storage_type =
      std::conditional_t<std::is_same_v<double, T>, __m256d,
                         std::conditional_t<std::is_same_v<float, T>, __m256, __m256i>>;

   using mask_storage_type = __m256i;
   using mask_lane_type = unsigned_lane_t<T>;

   // lane number
   static constexpr int size = N;
   static constexpr bool is_full = size == sizeof(__m256) / sizeof(T);
   static constexpr int width = 256;

   using isa_tag = avx2_isa_tag;
};

}  // namespace simd::details



#endif
