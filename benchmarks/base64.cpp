#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <print>
#include <span>
#include <string_view>
#include <system_error>

#include <simd.hpp>

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

constexpr std::size_t base64_encode_len(std::size_t input_len) {
   return (input_len + 2) / 3 * 4;
}

template <class Out>
struct base64_encode_result {
   Out out;
   std::size_t size;
};

constexpr std::size_t base64_decode_len(std::size_t input_len) {
   return (input_len / 4 * 3 + 2);
}

template <class Out>
struct base64_decode_result {
   Out out;
   std::size_t read_size;
   std::size_t written_size;
   std::errc ec;
};


namespace simd_base64 {



// simd_base64_encode()
//
// SIMD implementation of base64_endcode.
// Compared to our reference implementation we are about 2.1x faster on AVX2
// and 1.3x faster on SSE2.
// It is possible to reach an even faster implementation on AVX2 and SSE2, however this
// make the code more difficult to understand and merge several primitive together.
// For now, we think we have a good ratio readability / performance.
//
// The core algorithm is:
//   * Read 3 bytes
//   * Expand the 3 bytes to 4 (add a 0 byte padding at front)
//   * Convert big-endian to little-endian
//   * Extract the 4 sextets
//   * Encode the 4 sextets using the base 64 dictionary.
//   * Write the 4 encoded sextets to the output.
//
template <std::output_iterator<char> O>
   requires std::contiguous_iterator<O>
base64_encode_result<O> simd_base64_encode(std::span<const std::byte> input, O out) {

   using SimdU8 = simd::vec<std::uint8_t>;
   using SimdI8 = simd::vec<std::int8_t>;
   using SimdU32 = simd::rebind_t<std::uint32_t, SimdU8>;

   auto between = [](SimdI8 bytes, std::int8_t a, std::int8_t b) {
      return bytes >= a & bytes <= b;
   };

   // octets_to_sextets()
   //
   // Octets follow this pattern
   //    [00000000|aaaaaabb|bbbbcccc|ccdddddd]
   // will be regroup to form the 4 sextets with the following pattern
   //    [00dddddd|00cccccc|00bbbbbb|00aaaaaa]
   auto octets_to_sextets = [](SimdU8 octets) {
      // Some bits to regroup are splitted across differents bytes.
      // Working in the int32 domain with some bitmask is easier to read.

      SimdU32 octets_u32 = std::bit_cast<SimdU32>(octets);
      SimdU32 a = (octets_u32 & 0xFC0000) >> 18;
      SimdU32 b = (octets_u32 & 0x03F000) >> 4;
      SimdU32 c = (octets_u32 & 0x000FC0) << 10;
      SimdU32 d = (octets_u32 & 0x00003F) << 24;
      return std::bit_cast<SimdU8>(a | b | c | d);
   };

   // encode_sextets()
   //
   // Return the base 64 dictionary encoding of each sextet.
   auto encode_sextets = [&](SimdU8 sextets) {
      // Switching to signed domain make the codegen far better (at least on x86 where
      // comparison on uint8 is emulated).
      // As we do not use the full range, we are safe.
      // This simple change make me gain about 2x (417ns -> 276ns).
      //
      // I think the main performance lost between this algo and the other comes from
      // here. A perfect hash mapping should help a lot but make the code more difficult
      // to understand.
      //
      auto signed_sextets = std::bit_cast<SimdI8>(sextets);
      auto uppers = between(signed_sextets, 0, 25);
      auto lowers = between(signed_sextets, 26, 51);
      auto digits = between(signed_sextets, 52, 61);
      auto pluses = signed_sextets == 62;
      auto solidi = signed_sextets == 63;

      SimdI8 shifts = simd::select(uppers, SimdI8('A'), 0) |
                      simd::select(lowers, SimdI8('a' - 26), 0) |
                      simd::select(digits, SimdI8('0' - 52), 0) |
                      simd::select(pluses, SimdI8('+' - 62), 0) |
                      simd::select(solidi, SimdI8('/' - 63), 0);

      return sextets + std::bit_cast<SimdU8>(shifts);
   };

   const auto* src = input.data();
   std::size_t i = 0;
   constexpr std::size_t input_step = SimdU8::size / 4 * 3;
   for (; i + input_step < input.size(); i += input_step) {
      SimdU8 w(src + i);
      SimdU8 e = simd::expand(w, [](auto j) { return j % 4 != 0; });
      SimdU8 s = std::bit_cast<SimdU8>(simd::byteswap(std::bit_cast<SimdU32>(e)));
      SimdU8 sextets = octets_to_sextets(s);
      SimdU8 bytes = encode_sextets(sextets);
      bytes.copy_to(out);
      out += SimdU8::size;
   }
   // tail
   {
      auto remaining = input.size() - i;

      // Load only what is accessible
      // SimdU8 ramp_up{[](auto j) { return j; }};
      // auto ignore_last = ramp_up < remaining;
      // SimdU8 w{src + i, ignore_last};
      SimdU8 w = simd::partial_load<SimdU8>(src + i, remaining);

      // Apply the same algorithm but stop before write.
      SimdU8 e = simd::expand(w, [](auto j) { return j % 4 != 0; });
      SimdU8 s = std::bit_cast<SimdU8>(simd::byteswap(std::bit_cast<SimdU32>(e)));
      SimdU8 sextets = octets_to_sextets(s);
      SimdU8 bytes = encode_sextets(sextets);

      // Base 64 impose us to write a full word and partial ones should be fill with '='.
      // Due to the nature of the algorithm we could have 0, 1 or 2 '='.
      auto full_word = remaining / 3;
      auto partials = remaining % 3;
      auto partial_word = partials != 0;
      auto bytes_to_write = (full_word + partial_word) * 4;

      if (partials == 0) {
         // SimdU8::mask_type m{[](auto i) { return i < 4; }};
         // bytes.copy_to(out, m);
         std::memcpy(out, &bytes, bytes_to_write);
      }
      else {
         // In case of padding it is easier an faster to use a temporary buffer instead of
         // doing the same manipulaion on the simd register.
         std::array<std::uint8_t, SimdU8::size> buffer;
         bytes.copy_to(buffer.data());
         if (partials == 2) {
            buffer[bytes_to_write - 1] = '=';
         }
         else if (partials == 1) {
            buffer[bytes_to_write - 1] = '=';
            buffer[bytes_to_write - 2] = '=';
         }
         out = std::ranges::copy_n(buffer.begin(), bytes_to_write, out).out;
      }
   }

   return {};
}


std::string base64_encode(std::span<const std::byte> input) {
   std::string output;
   output.resize(base64_encode_len(input.size()));
   simd_base64_encode(input, output.data());
   return output;
}


// simd_base64_decode()
//
// SIMD implementation of base64_decode.
// Compared to our reference implementation we are about 2x/2.5x faster on AVX2
// and 1.5x/1.8x faster on SSE2.
// It is possible to reach an even faster implementation on AVX2 and SSE2, however this
// make the code more difficult to understand and merge several primitive together.
// For now, we think we have a good ratio readability / performance.
//
// The core algorithm is:
//   * Read at 4 ascii (input)
//   * Extract the 4 sextets from them (read_sextets)
//   * Convert the 4 sextets in the corresponding 3 octets (reorder_bytes)
//   * The 3 octets are big endian order, change that to little endian (byteswap)
//   * Remove the padding byte (compression)
//   * Store the 3 bytes to the output
//
template <std::output_iterator<std::byte> O>
   requires std::contiguous_iterator<O>
base64_decode_result<O> simd_base64_decode(std::string_view input, O out) {
   std::size_t len = input.size();
   if (len == 0) {
      return {std::move(out), 0, 0, {}};
   }
   if ((len < 4) || (len % 4) != 0) [[unlikely]] {
      return {std::move(out), 0, 0, std::errc::invalid_argument};
   }

   // Remove padding
   if (input[len - 1] == '=') {
      --len;
      if (input[len - 1] == '=') {
         --len;
      }
   }
   input = std::string_view(input.data(), len);

   using SimdU8 = simd::vec<std::uint8_t>;
   using SimdI8 = simd::rebind_t<std::int8_t, SimdU8>;
   using SimdU32 = simd::rebind_t<std::uint32_t, SimdU8>;

   auto between = [](SimdI8 bytes, std::int8_t a, std::int8_t b) {
      return bytes >= a & bytes <= b;
   };

   struct sextets_result {
      SimdU8 sextets;
      bool status;
   };

   // read_sextets()
   //
   // Process the base 64 dictionary.
   // Notes:
   // I think this code is readable for an SIMD code.
   // However, this is probably one of the bottleneck in term of performance improvement.
   // This pattern will consume a lot of registers and keep them alive for a "long" time.
   // Replacing this routine by a more performant one could make the overall
   // algorithm 1.85x faster.
   //
   auto read_sextets = [&](SimdU8 asciis) -> sextets_result {
      // As for encode, switching to signed integer domain make the code gen far better
      // (on x86 where unsigned comparison is emulated).
      // We get about 1.3x gain (394ns -> 295ns) with this simple change.
      // We are "safe" for the operation we do here.
      auto signed_asciis = std::bit_cast<SimdI8>(asciis);

      auto uppers = between(signed_asciis, 'A', 'Z');
      auto lowers = between(signed_asciis, 'a', 'z');
      auto digits = between(signed_asciis, '0', '9');
      auto pluses = signed_asciis == '+';
      auto solidi = signed_asciis == '/';

      bool ok = simd::all_of(uppers | lowers | digits | pluses | solidi);

      auto offsets = simd::select(uppers, SimdI8(-65), 0) |  //
                     simd::select(lowers, SimdI8(-71), 0) |  //
                     simd::select(digits, SimdI8(4), 0) |    //
                     simd::select(pluses, SimdI8(19), 0) |   //
                     simd::select(solidi, SimdI8(16), 0);    //
      return {asciis + std::bit_cast<SimdU8>(offsets), ok};
   };


   // reorder_bytes()
   //
   // Convert the 4 sextets in 3 octets + padding
   auto reorder_bytes = [](SimdU8 sextets) {
      // [00dddddd|00cccccc|00bbbbbb|00aaaaaa] x4
      //   0 1 2 3  4 5 6 7  8 9 A B  C D E F
      //
      // Target
      // [00000000|aaaaaabb|bbbbcccc|ccdddddd] x4
      //            D E F 9  A B 5 6  7 1 2 3
      //
      // input
      // [00dddddd|00cccccc|00bbbbbb|00aaaaaa] x4

      // [00000000|00cccccc|00000000|00aaaaaa] x4
      // even = std::simd_select(Mask{[](auto i) { return i % 2 == 0; }}, sextets, 0);
      // Faster way to extract even bytes:
      auto ca = std::bit_cast<SimdU32>(sextets) & SimdU32{0x003f003f};

      // [00dddddd|00000000|00bbbbbb|00000000] x4
      // odd = std::simd_select(Mask{[](auto i) { return i % 2 == 1; }}, sextets, 0);
      // Faster way to extract odd bytes:
      auto db = std::bit_cast<SimdU32>(sextets) & SimdU32{0x3f003f00};

      // [0000cccc|ccdddddd|0000aaaa|aabbbbbb] x4
      auto t0 = db >> 8 | ca << 6;

      // [dddd0000|aaaaaabb|bbbbcccc|ccdddddd] x4
      auto t1 = t0 >> 16 | t0 << 12;

      // [00000000|aaaaaabb|bbbbcccc|ccdddddd] x4
      return std::bit_cast<SimdU8>(t1 & 0x00FFFFFF);
   };

   // decoded_len()
   //
   // When we cannot process a full lane, we need to compute the number of octets base on
   // the readed sextets.
   auto decoded_len = [](std::size_t read_size) {
      auto mod_4 = read_size % 4;
      return read_size / 4 * 3 + (mod_4 - mod_4 / 2);
   };


   std::size_t i = 0;
   std::size_t written = 0;
   // I will have prefer to use std::ext::simd_for_each but the nature of the base 64
   // decoding make the last partial chunk a little bit annoying.
   // Indeed we need the number of readed bytes later and also fill the lanes with 'A'
   // instead of 0 to be able to reuse the existing primitives.

   const auto first = std::ranges::begin(input);
   for (; i + SimdU8::size < len; i += SimdU8::size) {
      SimdU8 w{first + i};
      auto [sextets, valid] = read_sextets(w);
      if (!valid) [[unlikely]]
         return {std::move(out), i, written, std::errc::invalid_argument};

      auto bytes = reorder_bytes(sextets);
      bytes = std::bit_cast<SimdU8>(simd::byteswap(std::bit_cast<SimdU32>(bytes)));
      bytes = simd::compress(bytes, [](auto j) { return j % 4 != 0; });

      written += SimdU8::size / 4 * 3;
      // safe only if overallocated but we have ensure that by construction.
      bytes.copy_to(reinterpret_cast<std::uint8_t*>(out));
      out += SimdU8::size / 4 * 3;
   }

   // tail
   {
      SimdU8 ramp_up{[](auto j) { return j; }};
      auto ignore_last = ramp_up < (len - i);
      SimdU8 w = simd::partial_load<SimdU8>(first + i, len - i);

      w = simd::select(ignore_last, w, std::uint8_t('A'));
      auto remaining_size = len - i;

      auto [sextets, valid] = read_sextets(w);
      if (!valid) [[unlikely]]
         return {std::move(out), i, written, std::errc::invalid_argument};
      auto bytes = reorder_bytes(sextets);

      bytes = std::bit_cast<SimdU8>(simd::byteswap(std::bit_cast<SimdU32>(bytes)));
      bytes = simd::compress(bytes, [](auto j) { return j % 4 != 0; });

      const auto w_len = decoded_len(remaining_size);
      written += w_len;
      // bytes.copy_to(reinterpret_cast<std::uint8_t*>(out));
      std::memcpy(out, &bytes, w_len);
      out += w_len;
   }

   return {std::move(out), len, written, {}};
}

std::vector<std::byte> base64_decode(std::string_view input) {
   std::vector<std::byte> output(base64_decode_len(input.size()));
   auto res = simd_base64_decode(input, output.data());
   if (res.ec != std::errc{})
      output.clear();
   else
      output.resize(res.written_size);
   return output;
}

}  // namespace simd_base64

namespace scalar {

static constexpr const char base64_enc_table[] =
   "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";


constexpr std::array<std::uint8_t, 256> make_dec_table() {
   std::array<std::uint8_t, 256> dec_table{0xFF};
   for (std::size_t i = 0; i < std::size(base64_enc_table); ++i)
      dec_table[base64_enc_table[i]] = static_cast<std::uint8_t>(i);
   dec_table['='] = 0;
   return dec_table;
}

static constexpr std::array<std::uint8_t, 256> base64_dec_table = make_dec_table();

template <std::output_iterator<char> O>
   requires std::contiguous_iterator<O>
base64_encode_result<O> base64_encode(std::span<const std::byte> input, O out) {
   std::size_t written = 0;
   std::size_t i = 0;
   std::size_t len = input.size();
   const std::byte* in = input.data();

   if (len > 3) {
      // load data 3 by 3 using a 32bits load.
      for (; i < len - 2; i += 3) {
         // Load the 3 bytes
         auto b1 = static_cast<std::uint8_t>(input[i]);
         auto b2 = static_cast<std::uint8_t>(input[i + 1]);
         auto b3 = static_cast<std::uint8_t>(input[i + 2]);

         // Compute the 4 indices from the 3 bytes
         auto i0 = b1 >> 2;
         auto i1 = ((b1 & 0x3) << 4) | ((b2 >> 4) & 0xF);
         auto i2 = ((b2 & 0xF) << 2) | ((b3 >> 6) & 0x3);
         auto i3 = b3 & 0x3F;

         // Write the 4 encoded bytes from the tables.
         *out++ = base64_enc_table[i0];
         *out++ = base64_enc_table[i1];
         *out++ = base64_enc_table[i2];
         *out++ = base64_enc_table[i3];
         written += 4;
      }
   }
   // We could have 2, 1, or 0 bytes left to encode.
   // Those are special case here.
   switch (len - i) {
   case 3: {
      // Load the 3 bytes
      auto b1 = static_cast<std::uint8_t>(input[i]);
      auto b2 = static_cast<std::uint8_t>(input[i + 1]);
      auto b3 = static_cast<std::uint8_t>(input[i + 2]);

      // Compute the 4 indices from the 3 bytes
      auto i0 = b1 >> 2;
      auto i1 = ((b1 & 0x3) << 4) | ((b2 >> 4) & 0xF);
      auto i2 = ((b2 & 0xF) << 2) | ((b3 >> 6) & 0x3);
      auto i3 = b3 & 0x3F;

      // Write the 4 encoded bytes from the tables.
      *out++ = base64_enc_table[i0];
      *out++ = base64_enc_table[i1];
      *out++ = base64_enc_table[i2];
      *out++ = base64_enc_table[i3];
   }
   case 2: {
      auto b1 = static_cast<std::uint8_t>(in[i]);
      auto b2 = static_cast<std::uint8_t>(in[i + 1]);
      *out++ = base64_enc_table[b1 >> 2];
      *out++ = base64_enc_table[((b1 & 0x3) << 4) | ((b2 >> 4) & 0xF)];
      *out++ = base64_enc_table[((b2 & 0xF) << 2)];
      *out++ = '=';
      written += 4;
      break;
   }
   case 1: {
      auto b1 = static_cast<std::uint8_t>(in[i]);
      *out++ = base64_enc_table[b1 >> 2];
      *out++ = base64_enc_table[((b1 & 0x3) << 4)];
      *out++ = '=';
      *out++ = '=';
      written += 4;
      break;
   }
   case 0:
      // nothing left.
   default:
      break;
   }
   return {std::move(out), written};
}

template <std::output_iterator<std::byte> O>
   requires std::contiguous_iterator<O>
base64_decode_result<O> base64_decode(std::string_view input, O out) {
   std::size_t len = input.size();
   if (len == 0) {
      return {std::move(out), 0, 0, {}};
   }
   if ((len < 4) || (len % 4) != 0) [[unlikely]] {
      return {std::move(out), 0, 0, std::errc::invalid_argument};
   }

   // Remove padding
   if (input[len - 1] == '=') {
      --len;
      if (input[len - 1] == '=') {
         --len;
      }
   }

   std::size_t written = 0;
   std::size_t quads = len / 4;
   auto src = input.begin();
   std::uint8_t block[4];
   for (std::size_t i = 0; i < quads; ++i, src += 4) {
      // Read 4 inputs byte and recompose them to 3 data bytes.
      // block[0] = 00111111
      // block[1] = 00112222
      // block[2] = 00222233
      // block[3] = 00333333
      block[0] = base64_dec_table[src[0]];
      block[1] = base64_dec_table[src[1]];
      block[2] = base64_dec_table[src[2]];
      block[3] = base64_dec_table[src[3]];
      if (((block[0] | block[1] | block[2] | block[3]) & 0xC0)) {
         // if high bits are set, we found and invalid input
         return {std::move(out), static_cast<std::size_t>(src - input.begin()), written,
                 std::errc::invalid_argument};
      }

      *out++ = static_cast<std::byte>((block[0] << 2) | (block[1] >> 4));
      *out++ = static_cast<std::byte>((block[1] << 4) | (block[2] >> 2));
      *out++ = static_cast<std::byte>((block[2] << 6) | block[3]);


      written += 3;
   }

   switch ((input.begin() + len) - src) {
   case 0:  // everything consumed
      break;
   case 1:  // impossible with padding
      return {std::move(out), static_cast<std::size_t>(src - input.begin()), written,
              std::errc::invalid_argument};
   case 2: {  // 2 input bytes -> 1 output byte
      block[0] = base64_dec_table[src[0]];
      block[1] = base64_dec_table[src[1]];
      if (((block[0] | block[1]) & 0xC0)) [[unlikely]] {
         // if high bits are set, we found and invalid input
         return {std::move(out), static_cast<std::size_t>(src - input.begin()), written,
                 std::errc::invalid_argument};
      }
      *out++ = static_cast<std::byte>((block[0] << 2) | (block[1] >> 4));
      src += 2;
      ++written;
      break;
   }
   case 3: {  // 3 input bytes -> 2 output bytes
      block[0] = base64_dec_table[src[0]];
      block[1] = base64_dec_table[src[1]];
      block[2] = base64_dec_table[src[2]];
      if (((block[0] | block[1] | block[2]) & 0xC0)) [[unlikely]] {
         // if high bits are set, we found and invalid input
         return {std::move(out), static_cast<std::size_t>(src - input.begin()), written,
                 std::errc::invalid_argument};
      }
      *out++ = static_cast<std::byte>((block[0] << 2) | (block[1] >> 4));
      *out++ = static_cast<std::byte>((block[1] << 4) | (block[2] >> 2));
      src += 3;
      written += 2;
      break;
   default:  // error?
      return {std::move(out), static_cast<std::size_t>(src - input.begin()), written,
              std::errc::invalid_argument};
   }
   }

   return {std::move(out), static_cast<std::size_t>(src - input.begin()), written, {}};
}

}  // namespace ref


namespace {

const char moby_dick_plain[] =
   "Call me Ishmael. Some years ago--never mind how long precisely--having\n"
   "little or no money in my purse, and nothing particular to interest me on\n"
   "shore, I thought I would sail about a little and see the watery part of\n"
   "the world. It is a way I have of driving off the spleen and regulating\n"
   "the circulation. Whenever I find myself growing grim about the mouth;\n"
   "whenever it is a damp, drizzly November in my soul; whenever I find\n"
   "myself involuntarily pausing before coffin warehouses, and bringing up\n"
   "the rear of every funeral I meet; and especially whenever my hypos get\n"
   "such an upper hand of me, that it requires a strong moral principle to\n"
   "prevent me from deliberately stepping into the street, and methodically\n"
   "knocking people's hats off--then, I account it high time to get to sea\n"
   "as soon as I can. This is my substitute for pistol and ball. With a\n"
   "philosophical flourish Cato throws himself upon his sword; I quietly\n"
   "take to the ship. There is nothing surprising in this. If they but knew\n"
   "it, almost all men in their degree, some time or other, cherish very\n"
   "nearly the same feelings towards the ocean with me.\n";

const char moby_dick_base64[] =
   "Q2FsbCBtZSBJc2htYWVsLiBTb21lIHllYXJzIGFnby0tbmV2ZXIgbWluZCBob3cgbG9uZ"
   "yBwcmVjaXNlbHktLWhhdmluZwpsaXR0bGUgb3Igbm8gbW9uZXkgaW4gbXkgcHVyc2UsIG"
   "FuZCBub3RoaW5nIHBhcnRpY3VsYXIgdG8gaW50ZXJlc3QgbWUgb24Kc2hvcmUsIEkgdGh"
   "vdWdodCBJIHdvdWxkIHNhaWwgYWJvdXQgYSBsaXR0bGUgYW5kIHNlZSB0aGUgd2F0ZXJ5"
   "IHBhcnQgb2YKdGhlIHdvcmxkLiBJdCBpcyBhIHdheSBJIGhhdmUgb2YgZHJpdmluZyBvZ"
   "mYgdGhlIHNwbGVlbiBhbmQgcmVndWxhdGluZwp0aGUgY2lyY3VsYXRpb24uIFdoZW5ldm"
   "VyIEkgZmluZCBteXNlbGYgZ3Jvd2luZyBncmltIGFib3V0IHRoZSBtb3V0aDsKd2hlbmV"
   "2ZXIgaXQgaXMgYSBkYW1wLCBkcml6emx5IE5vdmVtYmVyIGluIG15IHNvdWw7IHdoZW5l"
   "dmVyIEkgZmluZApteXNlbGYgaW52b2x1bnRhcmlseSBwYXVzaW5nIGJlZm9yZSBjb2Zma"
   "W4gd2FyZWhvdXNlcywgYW5kIGJyaW5naW5nIHVwCnRoZSByZWFyIG9mIGV2ZXJ5IGZ1bm"
   "VyYWwgSSBtZWV0OyBhbmQgZXNwZWNpYWxseSB3aGVuZXZlciBteSBoeXBvcyBnZXQKc3V"
   "jaCBhbiB1cHBlciBoYW5kIG9mIG1lLCB0aGF0IGl0IHJlcXVpcmVzIGEgc3Ryb25nIG1v"
   "cmFsIHByaW5jaXBsZSB0bwpwcmV2ZW50IG1lIGZyb20gZGVsaWJlcmF0ZWx5IHN0ZXBwa"
   "W5nIGludG8gdGhlIHN0cmVldCwgYW5kIG1ldGhvZGljYWxseQprbm9ja2luZyBwZW9wbG"
   "UncyBoYXRzIG9mZi0tdGhlbiwgSSBhY2NvdW50IGl0IGhpZ2ggdGltZSB0byBnZXQgdG8"
   "gc2VhCmFzIHNvb24gYXMgSSBjYW4uIFRoaXMgaXMgbXkgc3Vic3RpdHV0ZSBmb3IgcGlz"
   "dG9sIGFuZCBiYWxsLiBXaXRoIGEKcGhpbG9zb3BoaWNhbCBmbG91cmlzaCBDYXRvIHRoc"
   "m93cyBoaW1zZWxmIHVwb24gaGlzIHN3b3JkOyBJIHF1aWV0bHkKdGFrZSB0byB0aGUgc2"
   "hpcC4gVGhlcmUgaXMgbm90aGluZyBzdXJwcmlzaW5nIGluIHRoaXMuIElmIHRoZXkgYnV"
   "0IGtuZXcKaXQsIGFsbW9zdCBhbGwgbWVuIGluIHRoZWlyIGRlZ3JlZSwgc29tZSB0aW1l"
   "IG9yIG90aGVyLCBjaGVyaXNoIHZlcnkKbmVhcmx5IHRoZSBzYW1lIGZlZWxpbmdzIHRvd"
   "2FyZHMgdGhlIG9jZWFuIHdpdGggbWUuCg==";

void test_encode() {
   std::string_view input = moby_dick_plain;
   std::span bytes = std::as_bytes(std::span(input));
   std::string output = simd_base64::base64_encode(bytes);
   if (output != moby_dick_base64) {
      std::println("[FAILED] test_encode");
      std::exit(1);
   }
}

void test_decode() {
   auto output = simd_base64::base64_decode(moby_dick_base64);
   std::string_view result(reinterpret_cast<const char*>(output.data()), output.size());
   if (result != moby_dick_plain) {
      std::println("[FAILED] test_decode");
      std::exit(1);
   }
}

void bench_simd_base64_encode() {
   std::string_view input = moby_dick_plain;
   std::string output;
   output.resize(base64_encode_len(input.size()));

   ankerl::nanobench::Bench()
      .warmup(1500)
      .unit("element")
      .performanceCounters(true)
      .minEpochIterations(1500)
      .run("simd_base64_encode", [&] {
         std::span bytes = std::as_bytes(std::span(input));
         simd_base64::simd_base64_encode(bytes, output.data());
         ankerl::nanobench::doNotOptimizeAway(output);
      });
}



void bench_simd_base64_decode() {
   std::string_view input = moby_dick_base64;

   std::vector<std::byte> output(base64_decode_len(input.size()));
   ankerl::nanobench::Bench()
      .warmup(1500)
      .unit("element")
      .performanceCounters(true)
      .minEpochIterations(1500)
      .run("simd_base64_decode", [&] {
         auto res = simd_base64::simd_base64_decode(input, output.data());
         if (res.ec != std::errc{})
            output.clear();
         else
            output.resize(res.written_size);
         ankerl::nanobench::doNotOptimizeAway(output);
      });
}

void bench_scalar_base64_encode() {
   std::string_view input = moby_dick_plain;
   std::string output;
   output.resize(base64_encode_len(input.size()));

   ankerl::nanobench::Bench()
      .warmup(1500)
      .unit("element")
      .performanceCounters(true)
      .minEpochIterations(1500)
      .run("scalar_base64_encode", [&] {
         std::span bytes = std::as_bytes(std::span(input));
         scalar::base64_encode(bytes, output.data());
         ankerl::nanobench::doNotOptimizeAway(output);
      });
}



void bench_scalar_base64_decode() {
   std::string_view input = moby_dick_base64;

   std::vector<std::byte> output(base64_decode_len(input.size()));
   ankerl::nanobench::Bench()
      .warmup(1500)
      .unit("element")
      .performanceCounters(true)
      .minEpochIterations(1500)
      .run("scalar_base64_decode", [&] {
         auto res = scalar::base64_decode(input, output.data());
         if (res.ec != std::errc{})
            output.clear();
         else
            output.resize(res.written_size);
         ankerl::nanobench::doNotOptimizeAway(output);
      });
}


}  // namespace



int main(int argc, char** argv) {
   test_encode();
   test_decode();

   bench_simd_base64_encode();
   bench_simd_base64_decode();
   bench_scalar_base64_encode();
   bench_scalar_base64_decode();
   return 0;
}