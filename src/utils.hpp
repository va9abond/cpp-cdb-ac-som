#ifndef UTILS_HPP
#define UTILS_HPP


#include <iostream>
#include <fstream>
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
            std::cerr << "\n" << "====================================\n"
                      << msg
                      << "\n" << "====================================";
            std::exit(134);
        }
    }
}


namespace math {
    static const int32_t kMaxUlps = 4;

    inline bool almost_equal (double x, double y, int32_t maxUlps = kMaxUlps) {
        error_handler::_VERIFY(maxUlps > 0 && maxUlps < 4 * 1024 * 1024, "almost_equal: invalid maxUlps");

        int aInt = *(int*)&x;
        if (aInt < 0) // Make aInt lexicographically ordered as a twos-complement int
            aInt = 0x80000000 - aInt;

        int bInt = *(int*)&y;
        if (bInt < 0) // Make bInt lexicographically ordered as a twos-complement int
            bInt = 0x80000000 - bInt;

        int intDiff = abs(aInt - bInt);
        if (intDiff <= maxUlps)
            return true;

        return false;
    }

    inline bool is_double_grt (double x, double y) {
        return ( x > y + std::numeric_limits<double>::epsilon() );
    }
}


template <class Num_t = double>
inline std::vector<Num_t> operator+ ( const std::vector<Num_t>& rhs,
                                      const std::vector<Num_t>& lhs ) {
    error_handler::_VERIFY(rhs.size() == lhs.size(),
                           "vectors shouls have the same size");

    std::vector<Num_t> result (rhs.size());
    for (unsigned int i {0}; i < rhs.size(); ++i) {
        result[i] = rhs[i] + lhs[i];
    }

    return result;
}


template <class Num_t = double>
inline std::vector<Num_t> operator- ( const std::vector<Num_t>& rhs,
                                      const std::vector<Num_t>& lhs ) {
    error_handler::_VERIFY(rhs.size() == lhs.size(),
                           "vectors shouls have the same size");

    std::vector<Num_t> result (rhs.size());
    for (unsigned int i {0}; i < rhs.size(); ++i) {
        result[i] = rhs[i] - lhs[i];
    }

    return result;
}

template <class Num_t = double>
inline std::vector<Num_t>& operator+= ( std::vector<Num_t>& lhs,
                                        const std::vector<Num_t>& rhs ) {
    error_handler::_VERIFY(rhs.size() == lhs.size(),
                           "vectors shouls have the same size");
    for (unsigned int i {0}; i < lhs.size(); ++i) {
        lhs[i] += rhs[i];
    }

    return lhs;
}


template <class Num_t = double>
inline std::vector<Num_t> operator* ( const Num_t scal,
                                      const std::vector<Num_t> vec ) {
    std::vector<Num_t> result (vec.size());
    for (unsigned int i = 0; i < vec.size(); ++i) {
        result[i] = scal * vec[i];
    }

    return result;
}


namespace ccout {
    inline void print (const std::vector<char>& chars) {
        std::cout << "{ ";
        for (const auto& c : chars) {
            std::cout << (int)c << " ";
        }
        std::cout << "}\n";
    }

    inline void print (const std::vector<double>& doubles) {
        std::cout << "{ ";
        for (const auto& d : doubles) {
            std::cout << d << " ";
        }
        std::cout << "}\n";
    }

    inline void print (const alias::iipair& pair) {
        std::cout << std::format("({},{}) ", pair.first, pair.second);
    }
}


namespace vec_utils {
    template <class Valty>
    inline std::vector<double> normalize_vector (
            const std::vector<Valty>& vals
    ) {
        std::vector<double> result(vals.size());
        for (unsigned int i = 0; i < vals.size(); ++i) {
            // result[i] = (double)vals[i];
            result[i] = ((int)vals[i] == 0 ? 0.0 : 255.0);
        }

        return result;
    }
}


#endif // UTILS_HPP
