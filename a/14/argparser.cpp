#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>

#include "a/14/string_util.cpp"



class arg_parser
{
    std::vector<char*> args;
    std::vector<char*>::iterator args_it;

    void end_throw(const char* msg) const
    {
        if (end()) {
            throw std::invalid_argument(msg);
        }
    }

    void next()
    {
        args_it++;
    }

    char* get_arg() const
    {
        return *args_it;
    }

    char* get_arg_next()
    {
        return *(args_it++);
    }

    bool parse_arg(const char* short_s, const char* long_s)
    {
        char* arg = get_arg();
        if (std::strcmp(arg, short_s) != 0 && std::strcmp(arg, long_s) != 0) {
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

    bool load_arg_switch(bool& b, const char* short_s, const char* long_s, bool b_value = true)
    {
        end_throw("unexpected end of arguments");

        if (!parse_arg(short_s, long_s)) {
            return false;
        }

        b = b_value;
        return true;
    }

    template<typename T>
    bool load_num(T& n, const char* error_msg = nullptr)
    {
        end_throw(error_msg == nullptr ? "unexpected end of arguments" : error_msg);

        if (!str_to_num<T>(get_arg(), n)) {
            return false;
        }

        next();
        return true;
    }

    template<typename T>
    bool load_arg_num(T& n, const char* short_s, const char* long_s, const char* type_str = nullptr)
    {
        end_throw("unexpected end of arguments");

        if (!parse_arg(short_s, long_s)) {
            return false;
        }

        std::string arg_error_msg = format_string("%s / %s > %s argument expected", short_s, long_s, type_str == nullptr ? "numerical" : type_str);
        if (!load_num(n, arg_error_msg.data())) {
            throw std::invalid_argument(arg_error_msg);
        }

        return true;
    }

    // iterator elements have to be comparable with char*
    template<typename enum_type, typename iterator>
    bool load_arg_string_enum(enum_type& e, const char* short_s, const char* long_s, iterator options_first, iterator options_last, bool allow_index = false, const char* enum_options_str_o = nullptr)
    {
        end_throw("unexpected end of arguments");

        if (!parse_arg(short_s, long_s)) {
            return false;
        }

        std::string arg_error_msg = format_string("%s / %s > enum argument expected", short_s, long_s);
        if (enum_options_str_o != nullptr) {
            arg_error_msg += format_string(" (valid options : %s)", enum_options_str_o);
        }

        if (allow_index) {
            size_t e_index = 0;
            if (load_num(e_index, arg_error_msg.data()) && e_index < options_last - options_first) {
                e = static_cast<enum_type>(e_index);
                return true;
            }
        }

        end_throw(arg_error_msg.data());

        const char* arg = get_arg_next();
        iterator arg_it = std::find(options_first, options_last, arg);
        if (arg_it == options_last) {
            throw std::invalid_argument(arg_error_msg);
        }

        e = static_cast<enum_type>(arg_it - options_first);
        return true;
    }

    void throw_unknown_arg() const
    {
        throw std::invalid_argument(format_string("unknown argument - %s", get_arg()));
    }
};