#ifndef SIMD_DETAILS_ABIS_HPP
#define SIMD_DETAILS_ABIS_HPP


#include <simd/details/aggregated_abi.hpp>
#include <simd/details/emulated_abi.hpp>

#if defined(__x86_64__)
#include <simd/x86/x86_64_abi.hpp>
#elif defined(__ARM_NEON)
#include <simd/arm/neon_abi.hpp>
#endif


namespace simd {


namespace details {

template <class T>
using scalar_abi = details::emulated_abi<T, 1>;

}

#if defined(__AVX2__)

template <class T>
using native_abi = details::avx2_abi<T>;

template <class T, int N>
using current_abi = details::avx2_abi<T, N>;

template <class T, int N>
auto deduce_abi_f() {
   constexpr int native_size = native_abi<T>::size;
   if constexpr (N == 1) {
      return details::scalar_abi<T>{};
   }
   else if constexpr (N <= native_size / 2) {
      // sse
      return details::sse4_abi<T, N>{};
   }
   else if constexpr (N <= native_size) {
      // avx
      return details::avx2_abi<T, N>{};
   }
   // else if constexpr (register_size_target <= 512) {
   //    // avx512
   // }
   else {
      return details::aggregated_abi<T, N>{};
   }
}

#elif defined(__SSE4_2__)

template <class T>
using native_abi = details::sse4_abi<T>;

template <class T, int N>
using current_abi = details::sse4_abi<T, N>;

template <class T, int N>
auto deduce_abi_f() {
   constexpr int native_size = native_abi<T>::size;
   if constexpr (N == 1) {
      return details::scalar_abi<T>{};
   }
   else if constexpr (N < native_size) {
      // sse
      return details::sse4_abi<T, N>{};
   }
   else if constexpr (N == native_size) {
      // sse
      return details::sse4_abi<T, N>{};
   }
   // else if constexpr (register_size_target <= 512) {
   //    // avx512
   // }
   else {
      return details::aggregated_abi<T, N>{};
   }
}

#elif defined(__ARM_NEON)

template <class T>
using native_abi = details::neon_abi<T>;

template <class T, int N>
using current_abi = details::neon_abi<T, N>;


template <class T, int N>
auto deduce_abi_f() {
   constexpr int native_size = native_abi<T>::size;
   if constexpr (N == 1) {
      return details::scalar_abi<T>{};
   }
   else if constexpr (N <= native_size) {
      return details::neon_abi<T, N>{};
   }
   else {
      return details::aggregated_abi<T, N>{};
   }
}

#else

template <class T>
using native_abi = details::emulated_abi<T, 8>;

template <class T, int N>
using current_abi = details::emulated_abi<T, N>;


template <class T, int N>
auto deduce_abi_f() {
   return details::emulated_abi<T, N>{};
}

#endif


}  // namespace simd

#endif
