#ifndef SIMD_HPP
#define SIMD_HPP

// simd
//
// Toy implementation of `std::simd`
//


#include <simd/details/abis.hpp>
#include <simd/details/bit.hpp>
#include <simd/details/simd_fwd.hpp>
#include <simd/details/simd_shape.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>



namespace simd {



template <class T, int N>
struct deduce_abi {
   // For now, N should be less or equal to the native size
   // Later, we could have the aggregated abi.
   using type = decltype(deduce_abi_f<T, N>());
};



template <class T>
constexpr std::size_t mask_element_size_ = 0;

template <std::size_t Bytes>
using integer_from_ = void;

template <class T, simd_size_type_ N>
using deduce_t = typename deduce_abi<T, N>::type;

// alignement
template <class T, class U = typename T::value_type>
struct simd_alignment;

template <class T, class U = typename T::value_type>
inline constexpr std::size_t simd_alignment_v = simd_alignment<T, U>::value;



template <class... Flags>
struct simd_flags {};

inline constexpr simd_flags<> simd_flag_default{};
// simd_flag_convert
// simd_flag_aligned
// simd_flag_overaligned

template <class T, class Abi = native_abi<T>>
class basic_vec;

// N -> lane number
template <class T, int N = native_abi<T>::size>
using vec = basic_vec<T, deduce_t<T, N>>;

template <std::size_t Bytes, class Abi>
class basic_mask;

template <class T, int N = native_abi<T>::size>
using mask = basic_mask<sizeof(T), deduce_t<T, N>>;

template <class T, class V>
struct rebind {
   static constexpr int register_size = V::abi_type::width;
   static constexpr int src_size = V::size * sizeof(typename V::value_type);
   static constexpr int target_size = src_size / sizeof(T);

   using type = vec<T, target_size>;
};

template <int N, class V>
struct resize {
   using type = vec<typename V::value_tye, N>;
};



template <class T, class Abi>
class basic_vec {
   using Traits = Abi;

public:
   using value_type = T;
   using mask_type = basic_mask<sizeof(T), Abi>;
   using abi_type = Abi;

   // implementation
   using storage_type = typename Traits::value_storage_type;

   class reference {
   private:
      friend class basic_simd;

   public:
      reference() = delete;
      reference(const reference&) = delete;

      operator value_type() const noexcept {
         return Traits::get_lane(m_self, m_index);
      }

      template <class U>
      reference operator=(U&& value) && noexcept {
         Traits::set_lane(static_cast<value_type>(std::forward<U>(value)), m_self,
                          m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator+=(U&& value) && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         lane_value += static_cast<value_type>(std::forward<U>(value));
         Traits::set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator-=(U&& value) && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         lane_value -= static_cast<value_type>(std::forward<U>(value));
         Traits::set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator*=(U&& value) && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         lane_value *= static_cast<value_type>(std::forward<U>(value));
         Traits::set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator/=(U&& value) && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         lane_value /= static_cast<value_type>(std::forward<U>(value));
         Traits::set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator%=(U&& value) && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         lane_value %= static_cast<value_type>(std::forward<U>(value));
         Traits::set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator|=(U&& value) && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         lane_value |= static_cast<value_type>(std::forward<U>(value));
         Traits::set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator&=(U&& value) && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         lane_value &= static_cast<value_type>(std::forward<U>(value));
         Traits::set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator^=(U&& value) && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         lane_value ^= static_cast<value_type>(std::forward<U>(value));
         Traits::set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator<<=(U&& value) && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         lane_value <<= static_cast<value_type>(std::forward<U>(value));
         Traits::set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator>>=(U&& value) && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         lane_value >>= static_cast<value_type>(std::forward<U>(value));
         Traits::set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      reference operator++() && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         ++lane_value;
         Traits::set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      value_type operator++(int) && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         Traits::set_lane(lane_value + 1, m_self, m_index);
         return lane_value;
      }

      reference operator--() && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         --lane_value;
         Traits::set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      value_type operator--(int) && noexcept {
         auto lane_value = Traits::get_lane(m_self, m_index);
         Traits::set_lane(lane_value - 1, m_self, m_index);
         return lane_value;
      }


      friend void swap(reference&& lhs, reference&& rhs) noexcept {
         auto a = Traits::get_lane(lhs, lhs.m_index);
         auto b = Traits::get_lane(rhs, rhs.m_index);
         std::swap(a, b);
         Traits::set_lane(a, lhs, lhs.m_index);
         Traits::set_lane(b, rhs, rhs.m_index);
      }
      friend void swap(value_type& lhs, reference&& rhs) noexcept {
         auto b = Traits::get_lane(rhs, rhs.m_index);
         std::swap(lhs, b);
         Traits::set_lane(b, rhs, rhs.m_index);
      }
      friend void swap(reference&& lhs, value_type& rhs) noexcept {
         auto a = Traits::get_lane(lhs, lhs.m_index);
         std::swap(a, rhs);
         Traits::set_lane(a, lhs, lhs.m_index);
      }

   private:
      reference(basic_vec& self, std::size_t index)
         : m_self(self)
         , m_index(index) {}

      basic_vec& m_self;
      std::size_t m_index;
   };


   static constexpr std::integral_constant<int, abi_type::size> size{};

   // implementation details
   basic_vec(storage_type value)
      : m_data(value) {}

   operator storage_type const&() const& noexcept {
      return m_data;
   }

   operator storage_type&() & noexcept {
      return m_data;
   }

   operator storage_type() && noexcept {
      return m_data;
   }
   // end details

   constexpr basic_vec() noexcept = default;

   // [simd.ctor]
   template <class U, class From = std::remove_cvref_t<U>>
   basic_vec(U&& value) noexcept
      requires(std::convertible_to<From, T>)
      : m_data(Traits::template make<basic_vec>(value)) {}

   template <std::convertible_to<T> U, class UAbi>
   explicit basic_vec(const basic_vec<U, UAbi>& x) noexcept
      requires(basic_vec<U, UAbi>::size == size)
      : m_data(Traits::template convert<basic_vec>(x)) {}

   template <class G>
   explicit basic_vec(G&& gen) noexcept
      requires std::is_invocable_v<G, std::integral_constant<int, 0>>
      : m_data(Traits::template generate<basic_vec>(std::forward<G>(gen))) {}


   template <class It, class... Flags>
   basic_vec(It first, simd_flags<Flags...> = {})
      requires(std::contiguous_iterator<It>)
      : m_data(Traits::template load<basic_vec>(first)) {}

   template <class It, class... Flags>
   basic_vec(It first, const mask_type& mask, simd_flags<Flags...> = {})
      requires(std::contiguous_iterator<It>)
      : m_data(Traits::template load<basic_vec>(first, mask)) {}

   // [simd.copy]

   template <class It, class... Flags>
   void copy_from(It first, simd_flags<Flags...> = {})
      requires(std::contiguous_iterator<It>)
   {
      m_data = Traits::template load<basic_vec>(first);
   }

   template <class It, class... Flags>
   void copy_from(It first, const mask_type& mask, simd_flags<Flags...> = {})
      requires(std::contiguous_iterator<It>)
   {
      m_data = Traits::template load<basic_vec>(first, mask);
   }

   template <std::indirectly_writable<T> Out, class... Flags>
   void copy_to(Out out, simd_flags<Flags...> = {}) const
      requires(std::contiguous_iterator<Out>)
   {
      Traits::store(out, *this);
   }

   template <class Out, class... Flags>
   void copy_to(Out out, const mask_type& mask, simd_flags<Flags...> = {}) const {
      Traits::store(out, *this, mask);
   }

   // [simd.subscr]
   T operator[](std::size_t index) const& {
      return Traits::get_lane(*this, index);
   }

   reference operator[](std::size_t index) & {
      return {*this, index};
   }

   // [simd.unary]

   basic_vec& operator++() noexcept {
      return *this += basic_vec{1};
   }

   basic_vec operator++(int) noexcept {
      auto tmp = *this;
      ++*this;
      return tmp;
   }

   basic_vec& operator--() noexcept {
      return *this -= basic_vec{1};
   }

   basic_vec operator--(int) noexcept {
      auto tmp = *this;
      --*this;
      return tmp;
   }

   mask_type operator!() const noexcept {
      return *this != 0;
   }

   basic_vec operator~() noexcept {
      return Traits::bit_not(*this);
   }

   basic_vec operator+() noexcept {
      return *this;
   }

   basic_vec operator-() noexcept {
      return Traits::unary_minus(*this);
   }

   // [simd.binary]

   friend basic_vec operator+(const basic_vec& lhs, const basic_vec& rhs) {
      return Traits::add(lhs, rhs);
      // return details::add(Tag{}, lhs, rhs);
   }

   friend basic_vec operator-(const basic_vec& lhs, const basic_vec& rhs) {
      return Traits::sub(lhs, rhs);
   }

   friend basic_vec operator*(const basic_vec& lhs, const basic_vec& rhs) {
      return Traits::mul(lhs, rhs);
   }

   friend basic_vec operator/(const basic_vec& lhs, const basic_vec& rhs) {
      return Traits::div(lhs, rhs);
   }

   friend basic_vec operator%(const basic_vec& lhs, const basic_vec& rhs) {
      return Traits::rem(lhs, rhs);
   }

   friend basic_vec operator&(const basic_vec& lhs, const basic_vec& rhs) noexcept {
      return Traits::bit_and(lhs, rhs);
   }

   friend basic_vec operator|(const basic_vec& lhs, const basic_vec& rhs) noexcept {
      return Traits::bit_or(lhs, rhs);
   }

   friend basic_vec operator^(const basic_vec& lhs, const basic_vec& rhs) noexcept {
      return Traits::bit_xor(lhs, rhs);
   }

   friend basic_vec operator<<(const basic_vec& lhs, int rhs) noexcept {
      return Traits::bit_shift_left(lhs, rhs);
   }

   friend basic_vec operator>>(const basic_vec& lhs, int rhs) noexcept {
      return Traits::bit_shift_right(lhs, rhs);
   }

   // [simd.cassign]

   friend basic_vec& operator+=(basic_vec& lhs, const basic_vec& rhs) {
      return lhs = lhs + rhs;
   }

   friend basic_vec& operator-=(basic_vec& lhs, const basic_vec& rhs) {
      return lhs = lhs - rhs;
   }

   friend basic_vec& operator*=(basic_vec& lhs, const basic_vec& rhs) {
      return lhs = lhs * rhs;
   }

   friend basic_vec& operator/=(basic_vec& lhs, const basic_vec& rhs) {
      return lhs = lhs / rhs;
   }

   friend basic_vec& operator%=(basic_vec& lhs, const basic_vec& rhs) {
      return lhs = lhs % rhs;
   }

   friend basic_vec& operator&=(basic_vec& lhs, const basic_vec& rhs) {
      return lhs = lhs & rhs;
   }

   friend basic_vec& operator|=(basic_vec& lhs, const basic_vec& rhs) {
      return lhs = lhs | rhs;
   }

   friend basic_vec& operator^=(basic_vec& lhs, const basic_vec& rhs) {
      return lhs = lhs ^ rhs;
   }

   friend basic_vec& operator<<=(basic_vec& lhs, int rhs) {
      return lhs = lhs << rhs;
   }

   friend basic_vec& operator>>=(basic_vec& lhs, int rhs) {
      return lhs = lhs >> rhs;
   }

   // [simd.comparison]

   friend mask_type operator==(const basic_vec& lhs, const basic_vec& rhs) noexcept {
      return Traits::cmp_eq(lhs, rhs);
   }

   friend mask_type operator!=(const basic_vec& lhs, const basic_vec& rhs) noexcept {
      return !(lhs == rhs);
   }

   friend mask_type operator<(const basic_vec& lhs, const basic_vec& rhs) noexcept {
      return Traits::cmp_lt(lhs, rhs);
   }

   friend mask_type operator>(const basic_vec& lhs, const basic_vec& rhs) noexcept {
      return rhs < lhs;
   }

   friend mask_type operator<=(const basic_vec& lhs, const basic_vec& rhs) noexcept {
      return !(rhs < lhs);
   }

   friend mask_type operator>=(const basic_vec& lhs, const basic_vec& rhs) noexcept {
      return !(lhs < rhs);
   }

   // [simd.cond]

   friend basic_vec simd_select_impl(const mask_type& c,
                                     const basic_vec& a,
                                     const basic_vec& b) {
      return Traits::blend(c, b, a);
   }

private:
   storage_type m_data;
};


template <int N, class T, class Abi>
struct resize<N, basic_vec<T, Abi>> {
   using type = vec<T, N>;
};

template <std::size_t Bytes, class Abi>
class basic_mask {
   using Traits = Abi;

public:
   using value_type = bool;
   using abi_type = Abi;
   using storage_type = typename Traits::mask_storage_type;
   using mask_lane_type = typename Traits::mask_lane_type;

   class reference {
   private:
      friend class basic_simd_mask;

   public:
      reference() = delete;
      reference(const reference&) = delete;

      operator value_type() const noexcept {
         return Traits::mask_get_lane(m_self, m_index);
      }

      template <class U>
      reference operator=(U&& value) && noexcept {
         Traits::mask_set_lane(static_cast<value_type>(std::forward<U>(value)), m_self,
                               m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator+=(U&& value) && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         lane_value += static_cast<value_type>(std::forward<U>(value));
         Traits::mask_set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator-=(U&& value) && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         lane_value -= static_cast<value_type>(std::forward<U>(value));
         Traits::mask_set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator*=(U&& value) && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         lane_value *= static_cast<value_type>(std::forward<U>(value));
         Traits::mask_set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator/=(U&& value) && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         lane_value /= static_cast<value_type>(std::forward<U>(value));
         Traits::mask_set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator%=(U&& value) && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         lane_value %= static_cast<value_type>(std::forward<U>(value));
         Traits::mask_set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator|=(U&& value) && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         lane_value |= static_cast<value_type>(std::forward<U>(value));
         Traits::mask_set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator&=(U&& value) && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         lane_value &= static_cast<value_type>(std::forward<U>(value));
         Traits::mask_set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator^=(U&& value) && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         lane_value ^= static_cast<value_type>(std::forward<U>(value));
         Traits::mask_set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator<<=(U&& value) && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         lane_value <<= static_cast<value_type>(std::forward<U>(value));
         Traits::mask_set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      template <class U>
      reference operator>>=(U&& value) && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         lane_value >>= static_cast<value_type>(std::forward<U>(value));
         Traits::mask_set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      reference operator++() && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         ++lane_value;
         Traits::mask_set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      value_type operator++(int) && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         Traits::mask_set_lane(lane_value + 1, m_self, m_index);
         return lane_value;
      }

      reference operator--() && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         --lane_value;
         Traits::mask_set_lane(lane_value, m_self, m_index);
         return {m_self, m_index};
      }

      value_type operator--(int) && noexcept {
         auto lane_value = Traits::mask_get_lane(m_self, m_index);
         Traits::mask_set_lane(lane_value - 1, m_self, m_index);
         return lane_value;
      }


      friend void swap(reference&& lhs, reference&& rhs) noexcept {
         auto a = Traits::mask_get_lane(lhs, lhs.m_index);
         auto b = Traits::mask_get_lane(rhs, rhs.m_index);
         std::swap(a, b);
         Traits::mask_set_lane(a, lhs, lhs.m_index);
         Traits::mask_set_lane(b, rhs, rhs.m_index);
      }
      friend void swap(value_type& lhs, reference&& rhs) noexcept {
         auto b = Traits::mask_get_lane(rhs, rhs.m_index);
         std::swap(lhs, b);
         Traits::mask_set_lane(b, rhs, rhs.m_index);
      }
      friend void swap(reference&& lhs, value_type& rhs) noexcept {
         auto a = Traits::mask_get_lane(lhs, lhs.m_index);
         std::swap(a, rhs);
         Traits::mask_set_lane(a, lhs, lhs.m_index);
      }

   private:
      reference(basic_mask& self, std::size_t index)
         : m_self(self)
         , m_index(index) {}

      basic_mask& m_self;
      std::size_t m_index;
   };

   static constexpr std::integral_constant<int, abi_type::size> size{};

   // implementation details
   basic_mask(storage_type value)
      : m_data(value) {}

   operator storage_type const&() const& noexcept {
      return m_data;
   }

   operator storage_type&() & noexcept {
      return m_data;
   }

   operator storage_type() && noexcept {
      return m_data;
   }
   // end details

   basic_mask() = default;

   explicit basic_mask(bool value) noexcept
      : m_data(Traits::template make_mask<basic_mask>(value)) {}

   template <std::size_t UBytes, class UAbi>
   explicit basic_mask(const basic_mask<UBytes, UAbi>& x) noexcept
      requires(basic_mask<UBytes, UAbi>::size == size)
      : m_data(Traits::template convert_mask<basic_mask>(x)) {}

   template <class G>
   explicit basic_mask(G&& gen) noexcept
      requires(std::is_invocable_v<G, std::integral_constant<int, 0>>)
      : m_data(Traits::template generate_mask<basic_mask>(std::forward<G>(gen))) {}

   template <std::contiguous_iterator It, class... Flags>
   basic_mask(It first, simd_flags<Flags...> = {})
      requires(std::is_same_v<bool, std::iter_value_t<It>>)
      : m_data(Traits::template load_mask<basic_mask>(first)) {}

   template <std::contiguous_iterator It, class... Flags>
   basic_mask(It first, const basic_mask& mask, simd_flags<Flags...> = {})
      requires(std::is_same_v<bool, std::iter_value_t<It>>)
      : m_data(Traits::template load_mask<basic_mask>(first, mask)) {}

   template <std::contiguous_iterator It, class... Flags>
   void copy_from(It first, simd_flags<Flags...> = {})
      requires(std::is_same_v<bool, std::iter_value_t<It>>)
   {
      m_data = Traits::template load_mask<basic_mask>(first);
   }

   template <std::contiguous_iterator It, class... Flags>
   void copy_from(It first, const basic_mask& mask, simd_flags<Flags...> = {})
      requires(std::is_same_v<bool, std::iter_value_t<It>>)
   {
      m_data = Traits::template load_mask<basic_mask>(first, mask);
   }

   template <std::contiguous_iterator Out, class... Flags>
   void copy_to(Out out, simd_flags<Flags...> = {}) const
      requires(std::indirectly_writable<Out, bool>)
   {
      Traits::store_mask(out, *this);
   }

   template <std::contiguous_iterator Out, class... Flags>
   void copy_to(Out out, const basic_mask& mask, simd_flags<Flags...> = {}) const
      requires(std::indirectly_writable<Out, bool>)
   {
      Traits::store_mask(out, *this, mask);
   }

   bool operator[](std::size_t idx) const& {
      return Traits::mask_get_lane(*this, idx);
   }

   reference operator[](std::size_t idx) & {
      return {*this, idx};
   }

   basic_mask operator!() const noexcept {
      return Traits::mask_bit_not(*this);
   }

   basic_mask operator~() const noexcept {
      return Traits::mask_bit_not(*this);
   }

   friend basic_mask operator&&(const basic_mask& lhs, const basic_mask& rhs) noexcept {
      return Traits::mask_bit_and(lhs, rhs);  // shortcut?
   }

   friend basic_mask operator||(const basic_mask& lhs, const basic_mask& rhs) noexcept {
      return Traits::mask_bit_or(lhs, rhs);  // shortcut?
   }

   friend basic_mask operator&(const basic_mask& lhs, const basic_mask& rhs) noexcept {
      return Traits::mask_bit_and(lhs, rhs);
   }

   friend basic_mask operator|(const basic_mask& lhs, const basic_mask& rhs) noexcept {
      return Traits::mask_bit_or(lhs, rhs);
   }

   friend basic_mask operator^(const basic_mask& lhs, const basic_mask& rhs) noexcept {
      return Traits::mask_bit_xor(lhs, rhs);
   }

   friend basic_mask& operator&=(basic_mask& lhs, const basic_mask& rhs) {
      return lhs = lhs & rhs;
   }

   friend basic_mask& operator|=(basic_mask& lhs, const basic_mask& rhs) {
      return lhs = lhs | rhs;
   }

   friend basic_mask& operator^=(basic_mask& lhs, const basic_mask& rhs) {
      return lhs = lhs ^ rhs;
   }

   friend basic_mask operator==(const basic_mask& lhs, const basic_mask& rhs) noexcept {
      return Traits::logical_eq(lhs, rhs);
   }

   friend basic_mask operator!=(const basic_mask& lhs, const basic_mask& rhs) noexcept {
      return !(lhs == rhs);
   }

   // template <class T, class U>
   // friend auto simd_select_impl(const basic_simd_mask& c,
   //                                               const T& a,
   //                                               const T& b) noexcept;

private:
   storage_type m_data;
};



template <class V, class Abi>
auto simd_split(const basic_vec<typename V::value_type, Abi>& x) noexcept;

template <std::size_t Bytes, class Abi>
auto simd_split(const basic_mask<Bytes, Abi>& x) noexcept;

template <class T, class... Abis>
basic_vec<T, deduce_t<T, (basic_vec<T, Abis>::size + ...)>> simd_cat(
   const basic_vec<T, Abis>&...) noexcept;

template <std::size_t Bs, class... Abis>
basic_mask<Bs, deduce_t<integer_from_<Bs>, (basic_mask<Bs, Abis>::size() + ...)>>
simd_cat(const basic_mask<Bs, Abis>&...) noexcept;


// [simd.mask.reductions]

template <std::size_t Bytes, class Abi>
bool all_of(const basic_mask<Bytes, Abi>& k) noexcept {
   return Abi::all_of(k);
}

template <std::size_t Bytes, class Abi>
bool any_of(const basic_mask<Bytes, Abi>& k) noexcept {
   return !Abi::is_zero(k);
}

template <std::size_t Bytes, class Abi>
bool none_of(const basic_mask<Bytes, Abi>& k) noexcept {
   return Abi::is_zero(k);
}

template <std::size_t Bytes, class Abi>
simd_size_type_ reduce_count(const basic_mask<Bytes, Abi>& k) noexcept {
   return Abi::reduce_count(k);
}

template <std::size_t Bytes, class Abi>
simd_size_type_ reduce_min_index(const basic_mask<Bytes, Abi>& k) noexcept {
   if (none_of(k))
      return k.size;
   return Abi::reduce_min_index(k);
}

template <std::size_t Bytes, class Abi>
simd_size_type_ reduce_max_index(const basic_mask<Bytes, Abi>& k) noexcept {
   if (none_of(k))
      return -1;
   return Abi::reduce_max_index(k);
}


// overload on same_as<bool>??

// [simd.reductions]


// reduce
//
// Horizontal reduction by `op`.
template <class T, class Abi, class BinaryOperation = std::plus<>>
T reduce(const basic_vec<T, Abi>& x, BinaryOperation op = {})
   requires std::is_invocable_v<BinaryOperation, T, T>
{
   // by relaxing the spec from invocable<BinaryOperation, simd<T, 1>, simd<T, 1>> to
   // invocable<BinaryOperation, T, T>, it is easier to implements
   std::array<T, basic_vec<T, Abi>::size> hx;
   x.copy_to(hx.begin());
   return std::reduce(hx.begin() + 1, hx.end(), hx[0], std::move(op));
}

template <class T, class Abi, class BinaryOperation>
T reduce(const basic_vec<T, Abi>& x, const typename basic_vec<T, Abi>::mask_type& mask,
         T identity_element, BinaryOperation op)
   requires std::is_invocable_v<BinaryOperation, T, T>
{
   return reduce(simd_select(mask, x, identity_element), std::move(op));
}

template <class T, class Abi>
T reduce(const basic_vec<T, Abi>& x, const typename basic_vec<T, Abi>::mask_type& mask,
         std::plus<> op = {}) {
   return reduce(simd_select(mask, x, 0), op);
}

template <class T, class Abi>
T reduce(const basic_vec<T, Abi>& x, const typename basic_vec<T, Abi>::mask_type& mask,
         std::multiplies<> op = {}) {
   return reduce(simd_select(mask, x, 1), op);
}

template <class T, class Abi>
T reduce(const basic_vec<T, Abi>& x, const typename basic_vec<T, Abi>::mask_type& mask,
         std::bit_and<> op = {}) {
   return reduce(simd_select(mask, x, details::set_allbits<T>()), op);
}

template <class T, class Abi>
T reduce(const basic_vec<T, Abi>& x, const typename basic_vec<T, Abi>::mask_type& mask,
         std::bit_or<> op = {}) {
   return reduce(simd_select(mask, x, 0), op);
}

template <class T, class Abi>
T reduce(const basic_vec<T, Abi>& x, const typename basic_vec<T, Abi>::mask_type& mask,
         std::bit_xor<> op = {}) {
   return reduce(simd_select(mask, x, 0), op);
}


template <std::totally_ordered T, class Abi>
T reduce_min(const basic_vec<T, Abi>& x) {
   std::array<T, basic_vec<T, Abi>::size> hx;
   x.copy_to(hx.begin());
   return std::ranges::min(hx);
}


template <std::totally_ordered T, class Abi>
T reduce_max(const basic_vec<T, Abi>& x) {
   std::array<T, basic_vec<T, Abi>::size> hx;
   x.copy_to(hx.begin());
   return std::ranges::max(hx);
}

// [simd.alg]

template <class T, class Abi>
basic_vec<T, Abi> min(const basic_vec<T, Abi>& a, const basic_vec<T, Abi>& b) {
   return Abi::min(a, b);
}

template <class T, class Abi>
basic_vec<T, Abi> max(const basic_vec<T, Abi>& a, const basic_vec<T, Abi>& b) {
   return Abi::max(a, b);
}

template <class T, class Abi>
std::pair<basic_vec<T, Abi>, basic_vec<T, Abi>> minmax(const basic_vec<T, Abi>& a,
                                                       const basic_vec<T, Abi>& b) {
   return std::make_pair(Abi::min(a, b), Abi::max(a, b));
}

template <class T, class Abi>
basic_vec<T, Abi> clamp(const basic_vec<T, Abi>& v,
                        const basic_vec<T, Abi>& lo,
                        const basic_vec<T, Abi>& hi) {
   auto v_lo = select(v > lo, v, lo);
   return select(v < hi, v_lo, hi);
}

template <class T, class U>
auto select(bool c, const T& a, const U& b) -> std::remove_cvref_t<decltype(c ? a : b)> {
   return c ? a : b;
}

template <std::size_t Bytes, class Abi, class T, class U>
auto select(const basic_mask<Bytes, Abi>& c, const T& a, const U& b)
   -> decltype(simd_select_impl(c, a, b)) {
   return simd_select_impl(c, a, b);
}


template <std::size_t Bytes, class Abi, class T, class U>
auto simd_select_impl(const basic_mask<Bytes, Abi>& m, const T& a, const T& b)
   requires(std::same_as<bool, U>)
{
   return a ? m : !m;
}

template <std::size_t Bytes, class Abi, class T0, class T1,
          class U = std::common_type_t<T0, T1>>
auto simd_select_impl(const basic_mask<Bytes, Abi>& m, const T0& a, const T1& b)
   -> basic_vec<U, Abi>
   requires((sizeof(U) == Bytes) && std::convertible_to<T1, basic_vec<U, Abi>>)
{
   return select(m, basic_vec<U, Abi>{a}, basic_vec<U, Abi>{b});
}

template <simd_vec_type V>
constexpr V byteswap(const V& v) noexcept {
   using Abi = typename V::abi_type;
   return Abi::byteswap(v);
}

template <simd_vec_type V, class MaskGen>
constexpr V compress(const V& v, MaskGen g) noexcept {
   using Abi = typename V::abi_type;
   return Abi::compress(v, std::move(g));
}

template <simd_vec_type V, class MaskGen>
constexpr V expand(const V& v, MaskGen g) noexcept {
   using Abi = typename V::abi_type;
   return Abi::expand(v, std::move(g));
}

template <simd_vec_type V, std::contiguous_iterator I>
constexpr V partial_load(I first, std::iter_difference_t<I> n) {
   using Abi = typename V::abi_type;
   V x;
   Abi::partial_load(x, std::move(first), n);
   return x;
}

}  // namespace simd

#endif
