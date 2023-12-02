public:
    void update_step() {
        ++step;
        if (step % 500 == 0) { std::cout <<
            std::format("epoch: {} step: {}\n", epoch, step); }
    }

    void update_epoch() {
        ++epoch;
    }

    void update_ewidth() {
        ewidth = ( step > 1000 ?
                    ewidth0 * std::exp(-( (step) / (tau1) )) :
                    ewidth0 );
        sq_ewidth = ewidth * ewidth;
    }

    void update_lrate() { // [WARNING]: > for doubles
        lrate = lrate0 * std::exp(-( (step) / (tau2) ));
    }
