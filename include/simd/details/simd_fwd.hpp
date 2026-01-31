#ifndef SIMD_SIMD_FWD_HPP
#define SIMD_SIMD_FWD_HPP


#include <concepts>
#include <cstddef>

namespace simd {

using simd_size_type_ = int;

// rebind
template <class T, class V>
struct rebind;

template <class T, class V>
using rebind_t = typename rebind<T, V>::type;

// resize
template <int N, class V>
struct resize;

template <int N, class V>
using resize_t = typename resize<N, V>::type;

template <class T, class Abi>
class basic_vec;

template <std::size_t Bytes, class Abi>
class basic_mask;

template <class V>
concept simd_vec_type =
   std::same_as<V, basic_vec<typename V::value_type, typename V::abi_type>> &&
   std::is_default_constructible_v<V>;

// template<class V>
//   concept simd_mask_type =
//     std::same_as<V, basic_mask<mask-element-size<V>, typename V::abi_type>> &&
//     std::is_default_constructible_v<V>;

template <std::size_t Bytes, class Abi>
bool all_of(const basic_mask<Bytes, Abi>& k) noexcept;

template <std::size_t Bytes, class Abi>
bool any_of(const basic_mask<Bytes, Abi>& k) noexcept;

template <std::size_t Bytes, class Abi>
bool none_of(const basic_mask<Bytes, Abi>& k) noexcept;

template <std::size_t Bytes, class Abi>
simd_size_type_ reduce_count(const basic_mask<Bytes, Abi>& k) noexcept;

template <std::size_t Bytes, class Abi>
simd_size_type_ reduce_min_index(const basic_mask<Bytes, Abi>& k) noexcept;

template <std::size_t Bytes, class Abi>
simd_size_type_ reduce_max_index(const basic_mask<Bytes, Abi>& k) noexcept;

template <simd_vec_type V>
constexpr V byteswap(const V& v) noexcept;

}  // namespace simd

#endif
