#ifndef READ_MNIST_HPP
#define READ_MNIST_HPP


#include "utils.hpp"


// Thanks to Jayhello
// https://stackoverflow.com/a/52406407/18944269


namespace mnist {
    inline uint32_t swap_endian (uint32_t val) {
        val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
        return (val << 16) | (val >> 16);
    }


    inline void read_mnist (std::string file_path) {
        // std::ifstream file (file_path, std::ios::binary); // "t10k-images-idx3-ubyte.gz"
        std::ifstream file (file_path, std::ios::in | std::ios::binary);
        // "t10k-images-idx3-ubyte.gz"

        if (file.is_open()) {

            uint32_t magic = 0;
            uint32_t num_items = 0;
            uint32_t rows = 0;
            uint32_t cols = 0;

            file.read(reinterpret_cast<char*>(&magic), 4);
            magic = swap_endian(magic);
            error_handler::_VERIFY(magic == 2051, "Incorrect image file magic");

            file.read(reinterpret_cast<char*>(&num_items), 4);
            num_items = swap_endian(num_items);

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
#if 0
                std::vector<char> vector_chars (
                        pixels, pixels + rows*cols
                );
#endif
            }

            // std::vector<char> vector_chars (
            //         pixels, pixels + rows*cols
            // );
            //
            // ccout::print(vec_utils::convert_to_doubles(vector_chars));

            // std::cout << "{ ";
            // for (int i = 0; i < rows*cols; ++i) {
            //     std::cout << (int)pixels[i] << " ";
            // }

            // std::cout << "\n\n";


            // ccout::print(chars);

            // std::cout << " }\n";

            delete[] pixels;
        }
    }
}


#endif // READ_MNIST_HPP
