// find neuron-winner ix (min euclidean distance)
// for current input signal
const neuron& competition (
        const std::vector<double>& signal
    ) const {
    unsigned ix = neurons.size();
    double min_sq_dist = std::numeric_limits<double>::max();

    for (unsigned neuron_no {0}; neuron_no < neurons.size(); ++neuron_no) {
        double dist = sq_euclidean_distance(signal, neurons[neuron_no].weights);
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
    std::vector<neuron*> tpn {};
    double sq_e_width = ewidth * ewidth;
                                  // cooperation process
    for (unsigned i {0}; i < neurons.size(); ++i) {
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
    for (unsigned i {0}; i < tpn.size(); ++i) {
        // dw  = learning_rate * hjx * (sig - neuron.weights)
        double dist = neuron::distance(nw, neurons[i]);
        double hjix = std::exp(- ((dist * dist) / (2 * sq_e_width)) );
        vd dw = lrate * hjix * (sig - tpn[i]->weights);
        tpn[i]->weights += dw;
    }

    update_lrate();
    update_ewidth();
};
