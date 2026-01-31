#ifndef SIMD_STATIC_FOR_EACH_HPP
#define SIMD_STATIC_FOR_EACH_HPP

// Library utility to simulate a "template for".


#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>


namespace simd::details {

template <class F, class Tuple, std::size_t... Is>
constexpr auto static_for_each(Tuple&& tuple, F&& f, std::index_sequence<Is...>)
   -> decltype(std::forward<F>(f)) {
   using std::get;
   auto expander = {1, (std::invoke(f, get<Is>(std::forward<Tuple>(tuple))), 0)...};
   (void)expander;
   return std::forward<F>(f);
}

template <class F, class Tuple>
constexpr F static_for_each(Tuple&& tuple, F&& f) {
   ;
   return static_for_each(std::forward<Tuple>(tuple), std::forward<F>(f),
                          std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template <class F, class Tuple, std::size_t... Is>
constexpr auto static_for_each_index(Tuple&& tuple, F&& f, std::index_sequence<Is...>)
   -> decltype(std::forward<F>(f)) {
   using std::get;
   auto expander = {1, (std::invoke(f, get<Is>(std::forward<Tuple>(tuple)),
                                    std::integral_constant<std::size_t, Is>{}),
                        0)...};
   (void)expander;
   return std::forward<F>(f);
}

template <class F, class Tuple>
constexpr F static_for_each_index(Tuple&& tuple, F&& f) {
   return static_for_each_index(std::forward<Tuple>(tuple), std::forward<F>(f),
                                std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template <class F, std::size_t... Is>
constexpr static void for_template(F f, std::index_sequence<Is...>) {
   auto expander = {1, (f(std::integral_constant<int, Is>{}), 0)...};
   (void)expander;
}

} // namespace simd::details


#endif
