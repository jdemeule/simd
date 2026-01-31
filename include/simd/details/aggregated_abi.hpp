#ifndef SIMD_AGGREGATED_ABI_HPP
#define SIMD_AGGREGATED_ABI_HPP


#include <simd/details/emulated_abi.hpp>
#include <simd/details/simd_fwd.hpp>
#include <simd/details/simd_shape.hpp>
#include <simd/details/static_for_each.hpp>

#if defined(__x86_64__)
#include <simd/x86/arch.hpp>
#include <simd/x86/x86_64_abi.hpp>
#elif defined(__ARM_NEON)
#include <simd/arm/arch.hpp>
#include <simd/arm/neon_abi.hpp>
#endif

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>


namespace simd {

template <class T, class Abi>
class basic_vec;

template <class T, int N>
struct deduce_abi;

namespace details {

template <class F, class Tup>
struct tuple_transform {};

template <class F, class... Ts>
struct tuple_transform<F, std::tuple<Ts...>> {
   using type = std::tuple<typename F::template apply<Ts>::type...>;
};

struct extract_simd_size {
   template <class T>
   using apply = std::integral_constant<std::size_t, T::size>;
};

struct aggregated_isa_tag : public cpu_isa_tag {};



// template <class T, int N>
// constexpr auto aggregated_storage() {
//    constexpr int native_size = 256;
//    constexpr int target_size = sizeof(T) * N;
//    constexpr int words = target_size / native_size;
//    constexpr int remainings = target_size % native_size;
//    //basic_simd<T,
// }

template <class T, int N>
constexpr auto aggregated_storage() {
   constexpr int native_width = native_register_width();
   constexpr int native_size = native_width / __CHAR_BIT__ / sizeof(T);
   constexpr int target_size = N;
   constexpr int words = target_size / native_size;
   constexpr int remainings = target_size % native_size;
   static_assert(N >= 0);
   if constexpr (/*N == 0 ||*/ (words == 0 && remainings == 0))
      return std::make_tuple();
   else if constexpr (words == 0 && remainings > 0)
      return std::make_tuple(basic_vec<T, typename deduce_abi<T, remainings>::type>{});
   else
      return std::tuple_cat(
         std::make_tuple(basic_vec<T, typename deduce_abi<T, native_size>::type>{}),
         aggregated_storage<T, N - native_size>());
}

template <class T, int N>
constexpr auto aggregated_mask_storage() {
   constexpr int native_width = native_register_width();
   constexpr int native_size = native_width / __CHAR_BIT__ / sizeof(T);
   constexpr int target_size = N;
   constexpr int words = target_size / native_size;
   constexpr int remainings = target_size % native_size;
   if constexpr (words == 0 && remainings == 0)
      return std::make_tuple();
   else if constexpr (words == 0 && remainings > 0)
      return std::make_tuple(
         typename basic_vec<T, typename deduce_abi<T, remainings>::type>::mask_type{});
   else
      return std::tuple_cat(
         std::make_tuple(
            typename basic_vec<T,
                                typename deduce_abi<T, native_size>::type>::mask_type{}),
         aggregated_mask_storage<T, N - native_size>());
}

template <class Storage>
using compute_layout_t =
   decltype(std::tuple_cat(std::tuple<std::integral_constant<std::size_t, 0>>{},
                           typename tuple_transform<extract_simd_size, Storage>::type{}));

// template <class Storage>
// constexpr auto layout_of() {

//    std::array<
// }

template <class T, int N>
struct aggregated_abi {
   using value_type = T;
   using value_storage_type = decltype(aggregated_storage<T, N>());
   using mask_storage_type = decltype(aggregated_mask_storage<T, N>());
   using mask_lane_type = bool;
   // need the layout if I want to avoid full memcopy for get_lane/set_lane
   // but is it useful, as underlying will do it?

   using isa_tag = aggregated_isa_tag;

   static constexpr int size = N;

   static constexpr std::size_t aggregated_count = std::tuple_size_v<value_storage_type>;
   // static constexpr auto storage_layout = compute_layout<value_storage_type>();
   using storage_layout = compute_layout_t<value_storage_type>;

   template <int I, class TVec>
   static decltype(auto) get_leg(const TVec& x, std::integral_constant<int, I>) {
      return std::get<I>(static_cast<const value_storage_type&>(x));
   }
   template <int I, class TVec>
   static decltype(auto) get_leg(TVec& x, std::integral_constant<int, I>) {
      return std::get<I>(x);
   }

   template <int I, class TMask>
   static decltype(auto) get_mask_leg(const TMask& x, std::integral_constant<int, I>) {
      return std::get<I>(static_cast<const mask_storage_type&>(x));
   }
   template <int I, class TMask>
   static decltype(auto) get_mask_leg(TMask& x, std::integral_constant<int, I>) {
      return std::get<I>(static_cast<mask_storage_type&>(x));
   }

   template <int I>
   constexpr static auto leg_offset(std::integral_constant<int, I>) {
      if constexpr (I == 0)
         return 0;
      else {
         constexpr std::size_t offset = std::tuple_element_t<I, storage_layout>{};
         return offset + leg_offset(std::integral_constant<int, I - 1>{});
      }
   }

   template <class F>
   static void for_each_leg(F f) {
      for_template(std::move(f), std::make_index_sequence<aggregated_count>{});
   }

   template <class TVec>
   static auto make(typename TVec::value_type value) {
      value_storage_type storage;
      static_for_each(storage, [value](auto& x) { x = value; });
      return storage;
   }

   template <class TVec, class G>
   static auto generate(G&& gen) {
      value_storage_type storage;

      for_each_leg([&](auto i) {
         using Xi = std::tuple_element_t<i, value_storage_type>;
         constexpr std::size_t offset = leg_offset(i);
         auto& xi = get_leg(storage, i);
         xi = Xi{[&](auto ic) {
            using ic_offset = std::integral_constant<std::size_t, ic + offset>;
            return gen(ic_offset{});
         }};
      });
      return storage;
   }

   template <class TVec, class UVec>
   static TVec convert(const UVec& x) {
      return generate<TVec>([&](auto i) { return x[i]; });
   }

   template <class TVec, class It>
   static auto load(It it) {
      value_storage_type storage;
      static_for_each(storage, [&it](auto& x) {
         x.copy_from(it);
         it += x.size;
      });
      return storage;
   }

   template <class TVec, class TMask, class It>
   static auto load(It it, const TMask& m) {
      value_storage_type x;
      for_each_leg([&](auto idx) {
         const auto& mi = get_mask_leg(m, idx);
         auto& xi = get_leg(x, idx);
         xi.copy_from(it, mi);
         it += xi.size;
      });
      return x;
   }

   template <class TVec, class Out>
   static void store(Out out, const TVec& self) {
      for_each_leg([&](auto i) {
         const auto& xi = get_leg(self, i);
         xi.copy_to(out);
         out += xi.size;
      });
   }

   template <class TVec, class TMask, class Out>
   static void store(Out out, const TVec& self, const TMask& m) {
      for_each_leg([&](auto idx) {
         const auto& xi = get_leg(self, idx);
         const auto& mi = get_mask_leg(m, idx);
         xi.copy_to(out, mi);
         out += xi.size;
      });
   }

   template <class TVec, class F>
   static TVec apply_op(const TVec& lhs, const TVec& rhs, F op) {
      value_storage_type result;
      for_each_leg([&](auto idx) {
         const auto& xi = get_leg(lhs, idx);
         const auto& yi = get_leg(rhs, idx);
         get_leg(result, idx) = op(xi, yi);
      });
      return result;
   }

   template <class TVec>
   static TVec add(const TVec& lhs, const TVec& rhs) {
      return apply_op(lhs, rhs, [](const auto& x, const auto& y) { return x + y; });
   }
   template <class TVec>
   static TVec sub(const TVec& lhs, const TVec& rhs) {
      return apply_op(lhs, rhs, [](const auto& x, const auto& y) { return x - y; });
   }
   template <class TVec>
   static TVec mul(const TVec& lhs, const TVec& rhs) {
      return apply_op(lhs, rhs, [](const auto& x, const auto& y) { return x * y; });
   }

   template <class TVec>
   static TVec div(const TVec& lhs, const TVec& rhs) {
      return apply_op(lhs, rhs, [](const auto& x, const auto& y) { return x / y; });
   }

   template <class TVec>
   static TVec rem(const TVec& lhs, const TVec& rhs) {
      return apply_op(lhs, rhs, [](const auto& x, const auto& y) { return x % y; });
   }

   template <class TVec, class F>
   static TVec apply_unary_op(const TVec& lhs, F op) {
      value_storage_type result;
      for_each_leg([&](auto idx) {
         const auto& xi = get_leg(lhs, idx);
         get_leg(result, idx) = op(xi);
      });
      return result;
   }

   template <class TVec>
   static TVec unary_minus(const TVec& self) {
      return TVec{0} - self;
   }

   template <class TVec>
   static TVec bit_not(const TVec& self) {
      return apply_unary_op(self, [](const auto& x) { return ~x; });
   }

   template <class TVec>
   static TVec bit_and(const TVec& lhs, const TVec& rhs) {
      return apply_op(lhs, rhs, [](const auto& x, const auto& y) { return x & y; });
   }

   template <class TVec>
   static TVec bit_or(const TVec& lhs, const TVec& rhs) {
      return apply_op(lhs, rhs, [](const auto& x, const auto& y) { return x | y; });
   }

   template <class TVec>
   static TVec bit_xor(const TVec& lhs, const TVec& rhs) {
      return apply_op(lhs, rhs, [](const auto& x, const auto& y) { return x ^ y; });
   }

   template <class TVec>
   static TVec bit_shift_left(const TVec& lhs, int count) {
      return apply_unary_op(lhs, [count](const auto& x) { return x << count; });
   }

   template <class TVec>
   static TVec bit_shift_right(const TVec& lhs, int count) {
      return apply_unary_op(lhs, [count](const auto& x) { return x >> count; });
   }

   template <class TVec, class F>
   static auto apply_logical_op(const TVec& lhs, const TVec& rhs, F op) {
      mask_storage_type result;
      for_each_leg([&](auto idx) {
         const auto& xi = get_leg(lhs, idx);
         const auto& yi = get_leg(rhs, idx);
         get_leg(result, idx) = op(xi, yi);
      });
      return result;
   }

   template <class TVec>
   static mask_storage_type cmp_eq(const TVec& lhs, const TVec& rhs) {
      return apply_logical_op(
         lhs, rhs, [](const auto& x, const auto& y) { return x == y; });
   }

   template <class TVec>
   static auto cmp_lt(const TVec& lhs, const TVec& rhs) {
      return apply_logical_op(
         lhs, rhs, [](const auto& x, const auto& y) { return x < y; });
   }

   template <class TVec, class TMask>
   static auto blend(const TMask& m, const TVec& lhs, const TVec& rhs) {
      value_storage_type result;
      for_each_leg([&](auto idx) {
         const auto& mi = get_mask_leg(m, idx);
         const auto& xi = get_leg(lhs, idx);
         const auto& yi = get_leg(rhs, idx);
         // blend and simd_select invert parameters :/
         get_leg(result, idx) = simd_select(mi, yi, xi);
      });
      return result;
   }

   template <class TVec>
   static auto get_lane(const TVec& self, std::size_t index) {
      typename TVec::value_type data[TVec::size];
      self.copy_to(data);
      return data[index];
   }

   template <class TVec>
   static void set_lane(typename TVec::value_type value, TVec& self, std::size_t index) {
      typename TVec::value_type data[TVec::size];
      self.copy_to(data);
      data[index] = value;
      self.copy_from(data);
   }

   template <class TVecMask, class F>
   static auto mask_apply_logical_op(const TVecMask& lhs, const TVecMask& rhs, F op) {
      mask_storage_type result;
      for_each_leg([&](auto idx) {
         const auto& xi = get_mask_leg(lhs, idx);
         const auto& yi = get_mask_leg(rhs, idx);
         get_leg(result, idx) = op(xi, yi);
      });
      return result;
   }

   template <class TMask>
   static TMask logical_eq(const TMask& lhs, const TMask& rhs) {
      return mask_apply_logical_op(
         lhs, rhs, [](const auto& x, const auto& y) { return x == y; });
   }

   template <class TMask>
   static auto make_mask(bool value) {
      mask_storage_type storage;
      static_for_each(storage, [value](auto& x) {
         using SX = std::remove_cvref_t<decltype(x)>;
         using XAbi = typename SX::abi_type;
         x = XAbi::template make_mask<SX>(value);
      });
      return storage;
   }

   template <class TMask, class G>
   static TMask generate_mask(G&& gen) {
      mask_storage_type storage;

      for_each_leg([&](auto i) {
         using Xi = std::tuple_element_t<i, mask_storage_type>;
         constexpr std::size_t offset = std::tuple_element_t<i, storage_layout>{};
         auto& xi = get_mask_leg(storage, i);
         xi = Xi{[&](auto ic) {
            using ic_offset = std::integral_constant<std::size_t, ic + offset>;
            return gen(ic_offset{});
         }};
      });
      return storage;
   }

   template <class TMask, class UMask>
   static TMask convert_mask(const UMask& x) {
      return generate_mask<TMask>([&](auto i) { return x[i]; });
   }

   template <class TMask, class It>
   static auto load_mask(It it) {
      mask_storage_type storage;
      static_for_each(storage, [&it](auto& x) {
         x.copy_from(it);
         it += x.size;
      });
      return storage;
   }

   template <class TMask, class It>
   static auto load_mask(It it, const TMask& m) {
      mask_storage_type x;
      for_each_leg([&](auto idx) {
         const auto& mi = get_mask_leg(m, idx);
         auto& xi = get_mask_leg(x, idx);
         xi.copy_from(it, mi);
         it += xi.size;
      });
      return x;
   }

   template <class TMask, class Out>
   static void store_mask(Out out, const TMask& self) {
      for_each_leg([&](auto i) {
         const auto& xi = get_mask_leg(self, i);
         xi.copy_to(out);
         out += xi.size;
      });
   }

   template <class TMask, class Out>
   static void store_mask(Out out, const TMask& self, const TMask& m) {
      for_each_leg([&](auto idx) {
         const auto& xi = get_mask_leg(self, idx);
         const auto& mi = get_mask_leg(m, idx);
         xi.copy_to(out, mi);
         out += xi.size;
      });
   }

   template <class TMask>
   static bool mask_get_lane(const TMask& self, std::size_t index) {
      bool res = false;
      // std::size_t offset = 0;
      for_each_leg([&](auto i) {
         using Xi = std::tuple_element_t<i, mask_storage_type>;
         constexpr std::size_t offset = leg_offset(i);

         // constexpr std::size_t next_offset =
         //    leg_offset(std::integral_constant<int, i + 1>{});
         const auto& xi = get_mask_leg(self, i);
         constexpr std::size_t next_offset = offset + Xi::size;
         if (offset <= index && index < next_offset)
            res = xi[index - offset];
         // offset += xi.size;
      });
      return res;
   }

   template <class TMask>
   static void mask_set_lane(bool value, TMask& self, std::size_t index) {
      // std::size_t offset = 0;
      for_each_leg([&](auto i) {
         using Xi = std::tuple_element_t<i, mask_storage_type>;
         constexpr std::size_t offset = leg_offset(i);
         // constexpr std::size_t next_offset =
         //    leg_offset(std::integral_constant<int, i + 1>{});
         auto& xi = get_mask_leg(self, i);
         constexpr std::size_t next_offset = offset + Xi::size;
         if (offset <= index && index < (next_offset))
            xi[index - offset] = value;
         // offset += xi.size;
      });
   }

   template <class TMask>
   static TMask mask_bit_not(const TMask& self) {
      mask_storage_type result;
      for_each_leg([&](auto idx) {
         const auto& xi = get_mask_leg(self, idx);
         get_mask_leg(result, idx) = ~xi;
      });
      return result;
   }

   template <class TMask>
   static TMask mask_bit_and(const TMask& lhs, const TMask& rhs) {
      return mask_apply_logical_op(
         lhs, rhs, [](const auto& x, const auto& y) { return x & y; });
   }

   template <class TMask>
   static TMask mask_bit_or(const TMask& lhs, const TMask& rhs) {
      return mask_apply_logical_op(
         lhs, rhs, [](const auto& x, const auto& y) { return x | y; });
   }

   template <class TMask>
   static TMask mask_bit_xor(const TMask& lhs, const TMask& rhs) {
      return mask_apply_logical_op(
         lhs, rhs, [](const auto& x, const auto& y) { return x ^ y; });
   }


   template <class TMask>
   static bool all_of(const TMask& k) {
      bool res = true;
      for_each_leg([&](auto i) {
         const auto& ki = get_mask_leg(k, i);
         res &= simd::all_of(ki);
      });
      return res;
   }

   template <class TMask>
   static bool is_zero(const TMask& k) {
      bool res = true;
      for_each_leg([&](auto i) {
         const auto& ki = get_mask_leg(k, i);
         res &= simd::none_of(ki);
      });
      return res;
   }

   template <class TMask>
   static int reduce_count(const TMask& k) {
      int count = 0;
      for_each_leg([&](auto i) {
         const auto& ki = get_mask_leg(k, i);
         count += simd::reduce_count(ki);
      });
      return count;
   }

   template <class TMask>
   static int reduce_min_index(const TMask& k) {
      std::array<int, aggregated_count> indices;
      indices.fill(N);
      for_each_leg([&](auto i) {
         constexpr std::size_t offset = leg_offset(i);
         const auto& ki = get_mask_leg(k, i);
         auto index = simd::reduce_min_index(ki);
         if (index != ki.size)
            indices[i] = index + offset;
      });
      return std::ranges::min(indices);
   }

   template <class TMask>
   static int reduce_max_index(const TMask& k) {
      std::array<int, aggregated_count> indices;
      indices.fill(-1);
      for_each_leg([&](auto i) {
         constexpr std::size_t offset = leg_offset(i);
         const auto& ki = get_mask_leg(k, i);
         auto index = simd::reduce_max_index(ki);
         if (index != -1)
            indices[i] = index + offset;
      });
      return std::ranges::max(indices);
   }

   template <class TVec>
   static TVec byteswap(const TVec& self) {
      return apply_unary_op(self, [](const auto& x) { return simd::byteswap(x); });
   }

   template <class TVec, class MaskGen>
   static TVec compress(const TVec& x, MaskGen g) {
      constexpr std::size_t Sz = TVec::size;
      std::array<T, Sz> x_flat;
      x.copy_to(x_flat.begin());

      std::array<T, Sz> r_flat;
      std::size_t p = 0;
      for (std::size_t i = 0; i < N; ++i) {
         if (g(i))
            r_flat[p++] = x_flat[i];
      }
      return load(r_flat.begin());
   }

   template <class TVec, class MaskGen>
   static TVec expand(const TVec& x, MaskGen g) {
      constexpr std::size_t Sz = TVec::size;
      std::array<T, Sz> x_flat;
      x.copy_to(x_flat.begin());

      std::array<T, Sz> r_flat;
      std::size_t src = 0;
      for (std::size_t i = 0; i < N; ++i) {
         if (g(i))
            r_flat[i] = x_flat[src++];
      }
      return load(r_flat.begin());
   }

};

}  // namespace details

}  // namespace simd

#endif
