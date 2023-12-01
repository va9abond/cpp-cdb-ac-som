#ifndef SOKM_HPP
#define SOKM_HPP


#include "utils.hpp"

#define DEBUG_PRINT 0

struct neuron {
    using pair = alias::iipair;
    using vd   = alias::vd;


    neuron (int x, int y, vd&& ws) : coords({x,y}), weights(std::move(ws)) {}

    static double distance (const neuron& lhs, const neuron& rhs) {
        double sq_dist = 0;
        sq_dist = (rhs.coords.first - lhs.coords.first) *
                  (rhs.coords.first - lhs.coords.first)
                                    +
                  (rhs.coords.second - lhs.coords.second) *
                  (rhs.coords.second - lhs.coords.second);

        return std::sqrt(sq_dist);
    }


    const pair coords;
    vd weights;
};

inline bool operator== (const neuron& lhs, const neuron& rhs) {
    return (lhs.coords == rhs.coords);
}


class sokm {
    using ui = unsigned int;
    using vvd = alias::vvd;
    using vd = alias::vd;
    using vi = alias::vi;

public:
    sokm ( ui idim,
           ui fdim,
           ui n, ui m ):
        input_dim(idim),
        feature_dim(fdim),
        ewidth0( ((double)std::max(n,m)) / 2 ),
        step(0)
    {
        error_handler::_VERIFY(n*m == fdim, "incorrect feature dimension");

        update_ewidth();
        update_lrate();
        construct_feature_layer(n, m);
    }

private:
    // construct output neuron layer
    // [NOTE]: n*m shouldn't equals 0
    // ^
    // |
    // | m
    // |     n
    // + -------->
    void construct_feature_layer (int n, int m) {
        error_handler::_VERIFY((n*m) != 0 ,
                               "output layer shoud contain at least one dimension");

        for (ui x = 0; x < n; ++x) {
            for (ui y = 0; y < m; ++y) {
                neurons.push_back(
                        neuron( x,y,
                            construct_neuron_weights())
                );
            }
        }
    }

    vd construct_neuron_weights() {
        vd weights(input_dim, 0);

        for (auto& w : weights) {
            w = rndm::random<double>(-1.0, 1.0);
        }

        return weights;
    }

public:
    // square of euclidean distance
    static double sq_euclidean_distance (const vd& x, const vd& y) {
        error_handler::_VERIFY(x.size() == y.size(),
                               "vectors should have same dimensions");
        double sq_dist = 0;

        for (ui i {0}; i < x.size(); ++i) {
            sq_dist += (x[i] - y[i]) * (x[i] - y[i]);
        }

        return sq_dist;
    }

private:
    // find neuron-winner ix (min euclidean distance)
    // for current input signal
    neuron& competition (const vd& sig) {
        ui ix = neurons.size(); // no of neuron-winner
        double min_sq_dist = std::numeric_limits<double>::max(); // minimum square distance

                                      // competition process
        for (ui neuron_no {0}; neuron_no < neurons.size(); ++neuron_no) {
            double dist = sq_euclidean_distance(sig, neurons[neuron_no].weights);
            if (math::is_double_grt(min_sq_dist, dist)) {
                min_sq_dist = dist;
                ix = neuron_no;
            }
        }

        return neurons[ix];
    };

    // find topological neighbourhood
    // for neuron-winner (nw)
    std::vector<neuron*> cooperation (const neuron& nw) {
        // topological neighbourhood h_{j,i(x)}
        std::vector<neuron*> tpn {};
        double sq_e_width = ewidth * ewidth;
                                      // cooperation process
        for (ui i {0}; i < neurons.size(); ++i) {
            double dist = neuron::distance(nw, neurons[i]);
            double hjix = std::exp(- ((dist * dist) / (2 * sq_e_width)) );
            if (math::is_double_grt(hjix, 0)) {
                tpn.push_back(&neurons[i]);
            }
        }

        return tpn;
    };

    void adaptation (const vd& sig, const neuron& nw, const std::vector<neuron*>& tpn) {
        double sq_e_width = ewidth * ewidth;
                                       // adaptation process
        for (ui i {0}; i < tpn.size(); ++i) {
            // dw  = learning_rate * hjx * (sig - neuron.weights)
            double dist = neuron::distance(nw, neurons[i]);
            double hjix = std::exp(- ((dist * dist) / (2 * sq_e_width)) );
            vd dw = lrate * hjix * (sig - tpn[i]->weights);
            tpn[i]->weights += dw;
        }

        update_lrate();
        update_ewidth();
    };

public:
    void train (const vd& signal) {
        neuron& neuron_winner = competition(signal);

        // double sq_ewidth = ewidth * ewidth;
                                      // cooperation process
        for (auto& neuron : neurons) {
            const auto [dist, hjix] = hji(neuron_winner, neuron);
            if (math::is_double_grt(hjix, 0)) {
                                       // adaptation process (hjix > 0)
                // dw  = learning_rate * hjix * (signal - neuron.weights)
                vd dw = lrate * hjix * (signal - neuron.weights);
                neuron.weights += dw;
            }
        }

        updatae_constants();
// =====================================================
#if DEBUG_PRINT
        using std::cout;
        using std::right;

        printf("{\n");
        for (const auto& n : neurons) {
            printf("\t(%d,%d): { ", n.coords.first, n.coords.second);
            for (const auto& w : n.weights) {
                printf("%f ", w);
            }
            printf("};\n");
        }

        cout.width(50); cout << right << "===========\n";
        cout.width(50); cout << right <<
            std::format("lrate: {}\n", lrate);
        cout.width(50); cout << right <<
            std::format("ewidth: {}\n", ewidth);

        printf("}\n");
#endif
// =====================================================
    };

    void updatae_constants() {
        update_step();
        update_lrate();
        update_ewidth();
    }

    std::pair<double, double> hji (const neuron& n1, const neuron& n2) {
        double dist = neuron::distance(n1, n2);
        return {dist, std::exp(- ((dist * dist) / (2 * sq_ewidth)) )};
    }

    void update_step() { ++step; }

    void update_ewidth() {
        ewidth = ( step > 1000 ?
                    ewidth0 * std::exp(-( (step) / (tau1) )) :
                    ewidth0 );
        sq_ewidth = ewidth * ewidth;
    }

    void update_lrate() { // [WARNING]: > for doubles
        // lrate = ( math::is_double_grt(lrate0, lrate) ?
        //           lrate0 :
        //           lrate0 * std::exp(-( (step) / (tau2) )) );
        lrate = lrate0 * std::exp(-( (step) / (tau2) ));
    }


    const ui input_dim;
    const ui feature_dim;
    const double ewidth0;       // sigma0
    ui step;                    // learnin step, n
    const double lrate0 = 0.3;  // eta0 0.1
    const double tau1   = (1000 / std::log10(ewidth0)); // ewidth multiplier
    const double tau2   = 1000; // lrate multiplier
          double lrate  = 0;    // [TODO]: should be function
          double ewidth = 0;    // [TODO]: should be function effective width
          double sq_ewidth = ewidth * ewidth;
    std::vector<neuron> neurons; // vector of output(feature)
                                 // neuron layer
};


namespace ccout { // custom console output
    using std::cout;
    using std::right;

    inline void print (const neuron& n) {
        printf("\t(%d,%d): { ", n.coords.first, n.coords.second);
        for (const auto& w : n.weights) {
            printf("%f ", w);
        }
        printf("};\n");
    }

    inline void print (const sokm& map) {
        printf("{\n");
        for (const auto& n : map.neurons) {
            print(n);
        }

        cout.width(50); cout << right << "===========\n";
        cout.width(50); cout << right <<
            std::format("lrate: {}\n", map.lrate);
        cout.width(50); cout << right <<
            std::format("ewidth: {}\n", map.ewidth);

        printf("}\n");
    }
}


#endif // SOKM_HPP
