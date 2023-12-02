public:
    void train (const sdt::vector<double>& signal) {
                                      // competition process
        const neuron& neuron_winner = competition(signal);

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
    };
