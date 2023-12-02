inline bool operator== (const neuron& lhs, const neuron& rhs) {
    return (lhs.coords == rhs.coords);
}

inline bool operator< (const neuron& lhs, const neuron& rhs) {
    const auto& [x1,y1] = lhs.coords;
    const auto& [x2,y2] = rhs.coords;

    return (x1 == x2 ? y1 < y2 : x1 < x2);
}
