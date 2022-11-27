#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <stdexcept>
#include <optional>

#include "a/17/string_util.cpp"



class arg_parser
{
    std::vector<std::string_view> args;
    std::vector<std::string_view>::iterator args_it;

    void end_throw(std::string_view msg) const
    {
        if (end()) {
            throw std::invalid_argument(msg.data());
        }
    }

    void next()
    {
        args_it++;
    }

    std::string_view get_arg() const
    {
        return *args_it;
    }

    std::string_view get_arg_next()
    {
        return *(args_it++);
    }

    bool parse_arg(std::string_view short_s, std::string_view long_s)
    {
        std::string_view arg = get_arg();
        if (arg != short_s && arg != long_s) {
            return false;
        }

        next();
        return true;
    }

public:
    bool end() const
    {
        return args_it == args.end();
    }

    size_t size() const
    {
        return args.size();
    }

    arg_parser(int argc, char** argv) : args(argv + 1, argv + argc)
    {
        args_it = args.begin();
    }

    bool load_arg_switch(bool& b, std::string_view short_s, std::string_view long_s, bool b_value = true)
    {
        end_throw("unexpected end of arguments");

        if (!parse_arg(short_s, long_s)) {
            return false;
        }

        b = b_value;
        return true;
    }

    // tries loading number of type T
    // no arg    | error
    // no number | false
    // success   | true
    template<typename T>
    bool load_num(T& n, std::optional<std::string_view> error_msg = std::nullopt)
    {
        end_throw(error_msg.value_or("unexpected end of arguments"));

        std::optional<T> n_try = str_to_num<T>(get_arg());
        if (!n_try.has_value()) {
            return false;
        }

        next();
        n = n_try.value();
        return true;
    }

    template<typename T>
    bool load_arg_num(T& n, std::string_view short_s, std::string_view long_s, std::optional<std::string_view> type_str_o = std::nullopt)
    {
        end_throw("unexpected end of arguments");

        if (!parse_arg(short_s, long_s)) {
            return false;
        }

        std::string arg_error_msg = format_string("%s / %s > %s argument expected", short_s.data(), long_s.data(), type_str_o.value_or("numerical").data());
        if (!load_num(n, arg_error_msg)) {
            throw std::invalid_argument(arg_error_msg);
        }

        return true;
    }

    template<typename enum_type, typename iterator>
    bool load_arg_string_enum(enum_type& e, std::string_view short_s, std::string_view long_s, iterator options_first, iterator options_last, bool allow_index = false, std::optional<std::string_view> enum_options_str_o = std::nullopt)
    {
        end_throw("unexpected end of arguments");

        if (!parse_arg(short_s, long_s)) {
            return false;
        }

        std::string arg_error_msg = format_string("%s / %s > enum argument expected", short_s.data(), long_s.data());
        if (enum_options_str_o.has_value()) {
            arg_error_msg += format_string(" (valid options : %s)", enum_options_str_o.value().data());
        }

        if (allow_index) {
            size_t e_index = 0;
            if (load_num(e_index, arg_error_msg) && e_index < options_last - options_first) {
                e = static_cast<enum_type>(e_index);
                return true;
            }
        }

        end_throw(arg_error_msg);

        std::string_view arg = get_arg_next();
        iterator arg_it = std::find(options_first, options_last, arg);
        if (arg_it == options_last) {
            throw std::invalid_argument(arg_error_msg);
        }

        e = static_cast<enum_type>(arg_it - options_first);
        return true;
    }

    void throw_unknown_arg() const
    {
        throw std::invalid_argument(format_string("unknown argument - %s", get_arg().data()));
    }
};