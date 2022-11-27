#pragma once

#include <string>
#include <string_view>
#include <optional>
#include <stdexcept>
#include <charconv>

#include <memory>



template<typename ... Args>
std::string format_string(std::string_view format, Args ... args)
{
    int size = std::snprintf(nullptr, 0, format.data(), args ...);
    if (size < 0) {
        throw std::runtime_error("formatting error");
    }

    std::unique_ptr<char[]> sc = std::make_unique<char[]>(size + 1);
    int size_n = std::snprintf(sc.get(), size + 1, format.data(), args ...);
    if (size_n < 0 || size_n > size) {
        throw std::runtime_error("formatting error");
    }

    return std::string(sc.get()); // [todo] unnecessary char[] copying - use something like string + reserve / stringstream
}

template<typename T>
std::optional<T> str_to_num(std::string_view s)
{
    T value;
    const char* s_last = s.data() + s.size();
    auto [ptr, err] = std::from_chars(s.data(), s_last, value);

    if (ptr != s_last || err != std::errc()) {
        return std::nullopt;
    }

    return std::optional<T>(value);
}