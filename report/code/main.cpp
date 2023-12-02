#include "sokm.hpp"
#include "read_mnist.hpp"

int main() {

    sokm map (28*28, 10, 2, 5);
    sokm_education_mnist(map, "../data/train-images-idx3-ubyte", 2);
    const auto marks = sokm_check_mnist ( map,
            "../data/t10k-images-idx3-ubyte",
            "../data/t10k-labels-idx1-ubyte" );

    using std::cout;
    using std::right;
    using std::left;
    using std::format;

    std::vector<int> checker(10, 0);
    alias::ui fsize = 0;

    for (const auto& [coords, labels] : marks) {
        cout << format("({},{}) size: {}\n", coords.first,
                coords.second, labels.size());
        for (int l = 0; l < 5; ++l) {
            auto size1 = count(labels.begin(), labels.end(), l);
            auto size2 = count(labels.begin(), labels.end(), (9-l));

            cout << "\t"; cout.width(11);
            cout << left << format("[{}]: {}", l, size1);
            cout << left << format("[{}]: {}\n", (9-l), size2);

            fsize += size1; checker[l] += size1;
            fsize += size2; checker[9-l] += size2;
        }
    }

    cout << format("Total number: {}\n", fsize);
    for (alias::ui i = 0; i < 5; ++i) {
        cout << "\t"; cout.width(12);
        cout << left << format("[{}]: {}", i, checker[i]);
        cout << format("[{}]: {}\n", (9-i), checker[9-i]);
    }

    return 0;
}
