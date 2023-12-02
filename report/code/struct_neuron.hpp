struct neuron {
    neuron (int x, int y, std::vector<double>&& ws) :
        coords({x,y}),
        weights(std::move(ws))
    {}

    static double distance (const neuron& lhs, const neuron& rhs) {
        double sq_dist = 0;
        sq_dist = (rhs.coords.first - lhs.coords.first) *
                  (rhs.coords.first - lhs.coords.first)
                                    +
                  (rhs.coords.second - lhs.coords.second) *
                  (rhs.coords.second - lhs.coords.second);

        return std::sqrt(sq_dist);
    }


    const std::pair<int, int> coords;
    std::vector<double> weights;
}
