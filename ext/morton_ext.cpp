#include <cstdint>
#include <stdexcept>
#include <tuple>

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>

#include <libmorton/morton.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_morton, m) {
    m.doc() = "Scalar Morton 2D encoding/decoding via libmorton. "
              "Uses fastest available implementation "
              "(BMI2 > LUT fallback) selected at compile time.";

    // ---- 2D encode ----

    m.def("morton2D_32_encode",
        [](uint32_t x, uint32_t y) -> uint32_t {
            if (x > 0xFFFFu || y > 0xFFFFu)
                throw std::invalid_argument(
                    "x and y must be <= 65535 for 32-bit Morton encoding");
            return static_cast<uint32_t>(
                libmorton::morton2D_32_encode(
                    static_cast<uint_fast16_t>(x),
                    static_cast<uint_fast16_t>(y)));
        },
        "x"_a, "y"_a,
        "Encode 2D coordinates into a 32-bit Morton code.\n"
        "x and y must fit in 16 bits (0..65535).");

    m.def("morton2D_64_encode",
        [](uint32_t x, uint32_t y) -> uint64_t {
            return static_cast<uint64_t>(
                libmorton::morton2D_64_encode(
                    static_cast<uint_fast32_t>(x),
                    static_cast<uint_fast32_t>(y)));
        },
        "x"_a, "y"_a,
        "Encode 2D coordinates into a 64-bit Morton code.\n"
        "x and y must fit in 32 bits.");

    // ---- 2D decode ----

    m.def("morton2D_32_decode",
        [](uint32_t morton) -> std::tuple<uint32_t, uint32_t> {
            uint_fast16_t x{}, y{};
            libmorton::morton2D_32_decode(
                static_cast<uint_fast32_t>(morton), x, y);
            return {static_cast<uint32_t>(x), static_cast<uint32_t>(y)};
        },
        "morton"_a,
        "Decode a 32-bit Morton code into (x, y) coordinates.");

    m.def("morton2D_64_decode",
        [](uint64_t morton) -> std::tuple<uint32_t, uint32_t> {
            uint_fast32_t x{}, y{};
            libmorton::morton2D_64_decode(
                static_cast<uint_fast64_t>(morton), x, y);
            return {static_cast<uint32_t>(x), static_cast<uint32_t>(y)};
        },
        "morton"_a,
        "Decode a 64-bit Morton code into (x, y) coordinates.");
}
