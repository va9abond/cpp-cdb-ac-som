#ifndef NET_HPP
#define NET_HPP


#include "utils.hpp"


struct neuron {
    using pair = alias::iipair;
    using vd   = alias::vd;


    neuron (int x, int y) : coords({x,y}), weights() {}

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


struct sokm {
    using ui = unsigned int;
    using vvd = alias::vvd;
    using vd = alias::vd;
    using vi = alias::vi;


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
        construct_output_layer(n, m);
    }

    // construct output neuron layer
    // [NOTE]: n*m shouldn't equals 0
    // ^
    // |
    // | m
    // |     n
    // + -------->
    void construct_output_layer (int n, int m) {
        error_handler::_VERIFY((n*m) != 0 ,
                               "output layer shoud contain at least one dimension");

        for (ui x = 0; x < n; ++x) {
            for (ui y = 0; y < m; ++y) {
                neurons.push_back(neuron(x,y));
            }
        }
    }

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

    // find neuron-winner ix (min euclidean distance)
    // for current input signal
    neuron& competition (const vd& sig) {
        ui ix = neurons.size(); // no of neuron-winner
        double min_sq_dist = std::numeric_limits<double>::max(); // minimum square distance

                                      // competition process
        for (ui neuron_no {0}; neuron_no < neurons.size(); ++neuron_no) {
            double dist = sq_euclidean_distance(sig, neurons[neuron_no].weights);
            if (dist < min_sq_dist) {
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
            if (hjix > 0) { // [WARNING]: CHANEGE COMPARISON
                tpn.push_back(&neurons[i]);
            }
        }

        return tpn;
    };

    void adaptation (const vd& sig, const neuron& nw, const std::vector<neuron*>& tpn) {
        double sq_e_width = ewidth * ewidth;
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

    void train() {};

    void update_step() { ++step; }

    void update_ewidth() {
        ewidth = ( step > 1000 ?
                    ewidth0 * std::exp(-( (step) / (tau1) )) :
                    ewidth0 );
    }

    void update_lrate() { // [WARNING]: > for doubles
        lrate = ( lrate < lrate0 ?
                          lrate0 :
                          lrate0 * std::exp(-( (step) / (tau2) )) );
    }


    const ui input_dim;
    const ui feature_dim;
    const double ewidth0;      // sigma0
    ui step;                   // learnin step, n
    const double lrate0 = 0.3; // eta0 0.1
    const double tau1 = (1000 / std::log10(ewidth0));
    const double tau2 = 1000;
    double lrate = 0;          // [TODO]: should be function
    double ewidth = 0;         // [TODO]: should be function effective width
    std::vector<neuron> neurons; // vector of output(feature)
                                 // neuron layer
};


#endif // NET_HPP
