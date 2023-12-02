#include "sokm.hpp"
#include "read_mnist.hpp"


void sokm_education_mnist (sokm& map, std::string file_path) {
    std::ifstream file (file_path, std::ios::in | std::ios::binary);

    if (file.is_open()) {
        uint32_t magic = 0;
        uint32_t num_items = 0;
        uint32_t rows = 0;
        uint32_t cols = 0;

        file.read(reinterpret_cast<char*>(&magic), 4);
        magic = mnist::swap_endian(magic);
        error_handler::_VERIFY(magic == 2051, "incorrect image file magic");

        file.read(reinterpret_cast<char*>(&num_items), 4);
        num_items = mnist::swap_endian(num_items);

        file.read(reinterpret_cast<char*>(&rows), 4);
        rows = mnist::swap_endian(rows);

        file.read(reinterpret_cast<char*>(&cols), 4);
        cols = mnist::swap_endian(cols);

        std::cout << std::format("Images: {}\n", num_items);
        std::cout << std::format("\tRows: {}\n", rows);
        std::cout << std::format("\tCols: {}\n", cols);

        char* pixels = new char[rows * cols];

        for (int item = 0; item < num_items; ++item) {
            file.read(pixels, rows*cols);

            std::vector<char> vector_chars (
                    pixels, pixels + rows*cols
            );

            map.train(vec_utils::normalize_vector(vector_chars));
        }

        delete[] pixels;
    }
}


decltype(auto) sokm_check_mnist ( const sokm& map,
                       std::string images_path,
                       std::string labels_path ) {
    using alias::ui;

    std::ifstream images_file (images_path, std::ios::in | std::ios::binary);
    std::ifstream labels_file (labels_path, std::ios::in | std::ios::binary);

    std::map<alias::iipair, alias::vi> marks;
    for (const auto& n : map.neurons) {
        marks.emplace( n.coords, alias::vi{} );
    }

    if (images_file.is_open()) {
        uint32_t magic = 0;
        uint32_t num_items = 0;
        uint32_t num_labels = 0;
        uint32_t rows = 0;
        uint32_t cols = 0;

        images_file.read(reinterpret_cast<char*>(&magic), 4);
        magic = mnist::swap_endian(magic);
        error_handler::_VERIFY(magic == 2051,
                "incorrect image file magic");

        labels_file.read(reinterpret_cast<char*>(&magic), 4);
        magic = mnist::swap_endian(magic);
        error_handler::_VERIFY(magic == 2049,
                "incorrect label file magic");

        images_file.read(reinterpret_cast<char*>(&num_items), 4);
        num_items = mnist::swap_endian(num_items);

        labels_file.read(reinterpret_cast<char*>(&num_labels), 4);
        num_labels = mnist::swap_endian(num_labels);
        std::cout << std::format("labels: {}, items: {}\n",
                num_labels, num_items);
        error_handler::_VERIFY(num_items == num_labels,
                "numbers of labels and items doest't match");

        images_file.read(reinterpret_cast<char*>(&rows), 4);
        rows = mnist::swap_endian(rows);

        images_file.read(reinterpret_cast<char*>(&cols), 4);
        cols = mnist::swap_endian(cols);

        std::cout << std::format("Images: {}\n", num_items);
        std::cout << std::format("\tRows: {}\n", rows);
        std::cout << std::format("\tCols: {}\n", cols);

        char label_char;
        char* pixels = new char[rows * cols];

        for (int item = 0; item < num_items; ++item) {
            images_file.read(pixels, rows*cols);
            labels_file.read(&label_char, 1);
            int label = std::stoi(std::to_string(int(label_char)));

            std::vector<char> vector_chars (
                    pixels, pixels + rows*cols
            );

            auto coords = map.classify (
                    vec_utils::normalize_vector(vector_chars)
            );

            marks.at(coords).push_back(label);
        }

        delete[] pixels;
    }

    return marks;
}

int main() {

    sokm map (28*28, 10, 2, 5);
    sokm_education_mnist(map, "../data/train-images-idx3-ubyte");
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
        cout << format("({},{}) size: {}\n", coords.first, coords.second, labels.size());
        for (int l = 0; l < 5; ++l) {
            auto size1 = count(labels.begin(), labels.end(), l);
            auto size2 = count(labels.begin(), labels.end(), (9-l));

            cout << "\t"; cout.width(11); cout << left << format("[{}]: {}", l, size1);
            cout << left << format("[{}]: {}\n", (9-l), size2);

            fsize += size1; checker[l] += size1;
            fsize += size2; checker[9-l] += size2;
        }
    }

    cout << format("Total number: {}\n", fsize);
    for (alias::ui i = 0; i < 5; ++i) {
        cout << "\t"; cout.width(12); cout << left << format("[{}]: {}", i, checker[i]);
        cout << format("[{}]: {}\n", (9-i), checker[9-i]);
    }



    return 0;
}
