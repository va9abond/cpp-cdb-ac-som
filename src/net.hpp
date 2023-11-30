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
           ui n, ui m,
           double width0,
           double lrate0,
           double ewm, // effective width multiplier
           double lrm ) : // learning rate multiplier
        input_dim(idim),
        feature_dim(fdim),
        e_width0(width0),
        learning_rate0(lrate0),
        tau1(lrm),
        tau2(ewm),
        step(0)
    {
        error_handler::_VERIFY(n*m == fdim, "incorrect feature dimension");

        update_ewidth();
        update_learning_rate();
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
    std::vector<neuron*> cooperation (const neuron nw) {
        // topological neighbourhood h_{j,i(x)}
        std::vector<neuron*> tpn {};
        double sq_e_width = e_width * e_width;
                                      // cooperation process
        for (ui i {0}; i < neurons.size(); ++i) {
            double dist = neuron::distance(nw, neurons[i]);
            double hjix = std::exp(- ((dist * dist) / (2 * sq_e_width)) );
            if (hjix > 0) {
                tpn.push_back(&neurons[i]);
            }
        }

        return tpn;
    };

    // void adaptation (const vd& sig, const std::vector<neuron*>& tpn) {
    //     double sq_e_width = e_width * e_width;
    //     for (ui i {0}; i < tpn.size(); ++i) {
    //         // dw  = learning_rate * hjx * (sig - neuron.weights)
    //         double dist = neuron::distance(nw, neurons[i]);
    //         double hjix = std::exp(- ((dist * dist) / (2 * sq_e_width)) );
    //         // double dw = learning_rate *
    //     }
    //
    //     update_step();
    //     update_ewidth();
    // };

    void update_weights() {};

    void train() {};

    void update_step() { ++step; }

    void update_ewidth() {
        e_width = e_width0 * std::exp(- ((step) / (tau1)) );
    }

    void update_learning_rate() {
        learning_rate = learning_rate0 * std::exp(- ((step) / (tau2)) );
    }



    const ui input_dim;
    const ui feature_dim;
    const double e_width0; //
    const double learning_rate0;
    const double tau1;
    const double tau2;
    ui step;              // learnin step, n
    double learning_rate = 0;  // [TODO]: should be function
    double e_width = 0;        // [TODO]: should be function effective width
    std::vector<neuron> neurons; // vector of output(feature)
                                 // neuron layer
};


#endif // NET_HPP
