#ifndef SIMD_EMULATED_ABI_HPP
#define SIMD_EMULATED_ABI_HPP

#include <simd/details/arch.hpp>
#include <simd/details/simd_shape.hpp>
#include <simd/details/static_for_each.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <memory>
#include <utility>


namespace simd::details {

struct emulated_isa_tag : public cpu_isa_tag {};

template <class T, std::size_t N>
struct emulated_abi {
   using value_type = T;
   using value_storage_type = std::array<T, N>;
   using mask_storage_type = std::bitset<N>;
   using mask_lane_type = T;

   using isa_tag = emulated_isa_tag;

   static constexpr int size = N;

   template <class TVec>
   static TVec make(value_type value) {
      std::array<T, N> buffer;
      buffer.fill(value);
      return buffer;
   }


   template <class TVec, class G>
   static TVec generate(G&& gen) {
      std::array<T, N> buffer;
      for_template([&](auto ic) { buffer[ic] = gen(ic); }, std::make_index_sequence<N>{});
      return buffer;
   }

   template <class TVec, class UVec>
   static TVec convert(const UVec& x) {
      return generate<TVec>([&](auto i) { return x[i]; });
   }

   template <class TVec, class It>
   static auto load(It it) {
      std::array<T, N> buffer;
      std::copy_n(reinterpret_cast<const T*>(std::to_address(it)), N, buffer.begin());
      return buffer;
   }

   template <class TVec, class TMask, class It>
   static auto load(It it, const TMask& m) {
      std::array<T, N> buffer;
      for (std::size_t i = 0; i < N; ++i, ++it) {
         buffer[i] = m[i] ? *reinterpret_cast<const T*>(std::to_address(it)) : 0;
      }
      return buffer;
   }

   template <class TVec, class Out>
   static void store(Out out, const TVec& self) {
      const std::array<value_type, N>& datas = self;
      std::copy(datas.begin(), datas.end(), out);
   }

   template <class TVec, class TMask, class Out>
   static void store(Out out, const TVec& self, const TMask& m) {
      const std::array<value_type, N>& data = self;
      for (std::size_t i = 0; i < 16; ++i, ++out) {
         *out = m[i] ? data[i] : 0;
      }
   }

   template <class TVec>
   static TVec add(const TVec& lhs, const TVec& rhs) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = lhs[i] + rhs[i];
      }
      return buffer;
   }
   template <class TVec>
   static TVec sub(const TVec& lhs, const TVec& rhs) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = lhs[i] - rhs[i];
      }
      return buffer;
   }
   template <class TVec>
   static TVec mul(const TVec& lhs, const TVec& rhs) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = lhs[i] * rhs[i];
      }
      return buffer;
   }

   template <class TVec>
   static TVec div(const TVec& lhs, const TVec& rhs) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = lhs[i] / rhs[i];
      }
      return buffer;
   }

   template <class TVec>
   static TVec rem(const TVec& lhs, const TVec& rhs) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = lhs[i] % rhs[i];
      }
      return buffer;
   }

   template <class TVec>
   static TVec unary_minus(const TVec& self) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = -self[i];
      }
      return buffer;
   }

   template <class TVec>
   static TVec bit_not(const TVec& self) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = ~self[i];
      }
      return buffer;
   }

   template <class TVec>
   static TVec bit_and(const TVec& lhs, const TVec& rhs) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = lhs[i] & rhs[i];
      }
      return buffer;
   }

   template <class TVec>
   static TVec bit_or(const TVec& lhs, const TVec& rhs) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = lhs[i] | rhs[i];
      }
      return buffer;
   }

   template <class TVec>
   static TVec bit_xor(const TVec& lhs, const TVec& rhs) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = lhs[i] ^ rhs[i];
      }
      return buffer;
   }

   template <class TVec>
   static TVec bit_shift_left(const TVec& lhs, int count) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = lhs[i] << count;
      }
      return buffer;
   }

   template <class TVec>
   static TVec bit_shift_right(const TVec& lhs, int count) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = lhs[i] >> count;
      }
      return buffer;
   }

   template <class TVec>
   static mask_storage_type cmp_eq(const TVec& lhs, const TVec& rhs) {
      mask_storage_type m;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         m[i] = lhs[i] == rhs[i];
      }
      return m;
   }

   template <class TVec>
   static auto cmp_lt(const TVec& lhs, const TVec& rhs) {
      mask_storage_type m;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         m[i] = lhs[i] < rhs[i];
      }
      return m;
   }

   template <class TVec, class TMask>
   static auto blend(const TMask& m, const TVec& lhs, const TVec& rhs) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = m[i] ? lhs[i] : rhs[i];
      }
      return buffer;
   }

   template <class TVec>
   static auto get_lane(const TVec& self, std::size_t index) {
      const value_storage_type& data = self;
      return data[index];
   }

   template <class TVec>
   static void set_lane(value_type value, TVec& self, std::size_t index) {
      value_storage_type& data = self;
      data[index] = value;
   }

   template <class TVecMask>
   static TVecMask logical_eq(const TVecMask& lhs, const TVecMask& rhs) {
      const mask_storage_type& lhs_m = lhs;
      const mask_storage_type& rhs_m = rhs;

      mask_storage_type r;
      for (std::size_t i = 0; i < N; ++i) {
         r[i] = lhs_m[i] == rhs_m[i];
      }
      return r;
   }

   template <class TMask>
   static TMask make_mask(bool value) {
      return std::bitset<N>{value};
   }

   template <class TMask, class G>
   static TMask generate_mask(G&& gen) {
      std::bitset<N> k;
      for_template([&](auto ic) { k.set(ic, static_cast<bool>(gen(ic))); },
                   std::make_index_sequence<N>{});
      return k;
   }

   template <class TMask, class UMask>
   static TMask convert_mask(const UMask& x) {
      return generate_mask<TMask>([&](auto i) { return x[i]; });
   }

   template <class TMask, class It>
   static auto load_mask(It it) {
      std::bitset<N> data;
      for (int i = 0; i < TMask::size; ++i, ++it)
         data.set(i, *it);
      return data;
   }

   template <class TMask, class It>
   static auto load_mask(It it, const TMask& m) {
      std::bitset<N> data;
      for (int i = 0; i < TMask::size; ++i, ++it)
         data.set(i, m[i] ? *it : false);
      return data;
   }

   template <class TMask, class Out>
   static void store_mask(Out out, const TMask& self) {
      const std::bitset<N>& data = self;
      for (std::size_t i = 0; i < 16; ++i, ++out) {
         *out = data.test(i);
      }
   }

   template <class TMask, class Out>
   static void store_mask(Out out, const TMask& self, const TMask& m) {
      const std::bitset<N>& data = self;
      for (std::size_t i = 0; i < 16; ++i, ++out) {
         *out = m[i] ? data.test(i) : false;
      }
   }

   template <class TMask>
   static bool mask_get_lane(const TMask& self, std::size_t index) {
      const mask_storage_type& m = self;
      return m[index];
   }

   template <class TMask>
   static void mask_set_lane(bool value, TMask& self, std::size_t index) {
      mask_storage_type& m = self;
      m.set(index, value);
   }

   template <class TMask>
   static TMask mask_bit_not(const TMask& self) {
      mask_storage_type m = self;
      m.flip();
      return m;
   }

   template <class TMask>
   static TMask mask_bit_and(const TMask& lhs, const TMask& rhs) {
      mask_storage_type m = lhs;
      m &= static_cast<const mask_storage_type&>(rhs);
      return m;
   }

   template <class TMask>
   static TMask mask_bit_or(const TMask& lhs, const TMask& rhs) {
      mask_storage_type m = lhs;
      m |= static_cast<const mask_storage_type&>(rhs);
      return m;
   }

   template <class TMask>
   static TMask mask_bit_xor(const TMask& lhs, const TMask& rhs) {
      mask_storage_type m = lhs;
      m ^= static_cast<const mask_storage_type&>(rhs);
      return m;
   }


   template <class TMask>
   static bool all_of(const TMask& self) {
      const mask_storage_type& m = self;
      return m.all();
   }

   template <class TMask>
   static bool is_zero(const TMask& k) {
      const mask_storage_type& m = k;
      return m.none();
   }

   template <class TMask>
   static int reduce_count(const TMask& k) {
      const mask_storage_type& m = k;
      return m.count();
   }

   template <class TMask>
   static int reduce_min_index(const TMask& k) {
      const mask_storage_type& m = k;

      for (int i = 0; i < m.size(); ++i)
         if (m.test(i))
            return i;

      return m.size();
   }

   template <class TMask>
   static int reduce_max_index(const TMask& k) {
      const mask_storage_type& m = k;

      for (int i = m.size() - 1; i > 0; --i)
         if (m.test(i))
            return i;

      return -1;
   }

   template <class TVec>
   static TVec byteswap(const TVec& x) {
      std::array<value_type, TVec::size> buffer;
      for (std::size_t i = 0; i < TVec::size; ++i) {
         buffer[i] = std::byteswap(x[i]);
      }
      return buffer;
   }

   template <class TVec, class MaskGen>
   static TVec compress(const TVec& x, MaskGen g) {
      constexpr std::size_t Sz = TVec::size;

      std::array<T, Sz> r;
      std::size_t p = 0;
      for (std::size_t i = 0; i < N; ++i) {
         if (g(i))
            r[p++] = x[i];
      }
      return r;
   }

   template <class TVec, class MaskGen>
   static TVec expand(const TVec& x, MaskGen g) {
      constexpr std::size_t Sz = TVec::size;
      std::array<T, Sz> r;
      std::size_t src = 0;
      for (std::size_t i = 0; i < N; ++i) {
         if (g(i))
            r[i] = x[src++];
      }
      return r;
   }
};


}  // namespace simd::details



#endif
