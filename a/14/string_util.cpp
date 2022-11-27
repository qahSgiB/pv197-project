#pragma once

#include <string>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <cstdio>
#include <cstdlib>



template<typename ... Args>
std::string format_string(const char* format, Args ... args)
{
    int size = std::snprintf(nullptr, 0, format, args ...);
    if (size < 0) {
        throw std::runtime_error("formatting error");
    }

    std::unique_ptr<char[]> sc = std::make_unique<char[]>(size + 1);
    int size_n = std::snprintf(sc.get(), size + 1, format, args ...);
    if (size_n < 0 || size_n > size) {
        throw std::runtime_error("formatting error");
    }

    return std::string(sc.get()); // [todo] unnecessary char[] copying - use something like string + reserve / stringstream
}

template<typename T>
std::enable_if_t<std::is_signed<T>::value, bool> str_to_num(const char* s, T& value)
{
    char* s_last;
    long long value_ll = std::strtoll(s, &s_last, 10); // not optimized for smaller types

    if (*s_last != '\0' || value_ll < static_cast<long long>(std::numeric_limits<T>::lowest()) || value_ll > static_cast<long long>(std::numeric_limits<T>::max())) {
        return false;
    }

    value = static_cast<T>(value_ll);
    return true;
}

template<typename T>
std::enable_if_t<std::is_unsigned<T>::value, bool> str_to_num(const char* s, T& value)
{
    char* s_last;
    unsigned long long value_ll = std::strtoull(s, &s_last, 10); // not optimized for smaller types

    if (*s_last != '\0' || value_ll > static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
        return false;
    }

    value = static_cast<T>(value_ll);
    return true;
}