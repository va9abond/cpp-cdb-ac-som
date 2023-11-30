#ifndef UTILS_HPP
#define UTILS_HPP


#include <iostream>
#include <cmath>
#include <utility>
#include <map>
#include <set>
#include <vector>
#include <queue>
#include <utility>
#include <string>
#include <random>
#include <format>
#include <algorithm>
#include <functional>


namespace alias {
    using ui = unsigned int;

    using vvd = std::vector<std::vector<double>>;
    using vd  = std::vector<double>;
    using vvi = std::vector<std::vector<int>>;
    using vi  = std::vector<int>;

    using iipair = std::pair<int,int>;
}

namespace rndm {

    template <class Num_t, typename Gen_t = std::mt19937>
    Num_t random (const Num_t lower_bound, const Num_t upper_bound) {
        static Gen_t generator(std::random_device{}());

        using unfdist_type = typename std::conditional <
            std::is_integral<Num_t>::value,
            std::uniform_int_distribution<Num_t>,
            std::uniform_real_distribution<Num_t>
        >::type;

        static unfdist_type unfdist;
        return unfdist(
                generator,
                typename unfdist_type::param_type{lower_bound, upper_bound});
    }

}


namespace error_handler {
    inline void _VERIFY (bool expression, const char* msg) {
        if (expression == false)
        {
            std::cerr << "\n" << "===================================="
                      << msg
                      << "\n" << "====================================";
            std::exit(134);
        }
    }
}

template <class Num_t = double>
std::vector<Num_t> operator+ ( const std::vector<Num_t>& rhs,
                               const std::vector<Num_t>& lhs ) {
    error_handler::_VERIFY(rhs.size() == lhs.size());

    std::vector<Num_t> result (rhs.size());
    for (unsigned int i {0}; i < rhs.size(); ++i) {
        result[i] = rhs[i] + lhs[i];
    }

    return result;
}


template <class Num_t = double>
std::vector<Num_t> operator- ( const std::vector<Num_t>& rhs,
                               const std::vector<Num_t>& lhs ) {
    error_handler::_VERIFY(rhs.size() == lhs.size());

    std::vector<Num_t> result (rhs.size());
    for (unsigned int i {0}; i < rhs.size(); ++i) {
        result[i] = rhs[i] - lhs[i];
    }

    return result;
}


template <class Num_t = double>
std::vector<Num_t> operator* ( const Num_t scal,
                               const std::vector<Num_t> vec ) {
    std::vector<Num_t> result (vec.size());
    for (unsigned int i = 0; i < vec.size(); ++i) {
        result[i] = scal * vec[i];
    }

    return result;
}


#endif // UTILS_HPP
