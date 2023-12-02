class sokm {
public:
    sokm ( unsigned int idim,
           unsigned int fdim,
           unsigned int n, unsigned int m ):
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
    void construct_feature_layer (int n, int m) {
        error_handler::_VERIFY((n*m) != 0 ,
                               "output layer shoud contain at least one dimension");

        for (unsigned x = 0; x < n; ++x) {
            for (unsigned y = 0; y < m; ++y) {
                neurons.push_back(
                        neuron( x, y,
                            construct_neuron_weights())
                );
            }
        }
    }

    std::vector<double> construct_neuron_weights() {
        std::vector<double> weights(input_dim, 0);

        for (auto& w : weights) {
            w = rndm::random<double>(-1.0, 1.0);
        }

        return weights;
    }


    const unsigned int input_dim;
    const unsigned int feature_dim;
    const double       ewidth0;
          unsigned int step;
          unsigned int epoch = 1;
    const double       lrate0 = 0.3;
    const double       tau1 = (1000 / std::log10(ewidth0));
    const double       tau2 = 1000;
          double       lrate  = 0;
          double       ewidth = 0;
          double       sq_ewidth = ewidth * ewidth;
    std::vector<neuron> neurons;
};
