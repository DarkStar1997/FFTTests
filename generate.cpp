#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <fmt/core.h>
#include <fmt/format.h>

int main()
{
    size_t n; std::cin >> n;
    constexpr size_t buffer_size = 1024 * 1024;
    const std::string filename = "test_data";

    std::string buffer; buffer.reserve(buffer_size + 100);
    buffer += std::to_string(n); buffer += '\n';

    std::mt19937_64 rng{std::random_device()()};
    std::uniform_int_distribution<int> dis(1, 100);
    std::ofstream out; out.open(filename);
    fmt::print("Generating and writing {} random numbers between {} and {} to {}\n", n, 1, 100, filename);

    while(n--) {
        buffer += fmt::format_int(dis(rng)).str(); buffer += '\n';
        if(buffer.length() >= buffer_size) {
            out.write(buffer.data(), buffer.length());
            buffer.clear();
        }
    }

    if(buffer.length() > 0) {
        out.write(buffer.data(), buffer.length());
    }

    out.close();
    fmt::print("Test data generated and written to {}\n", filename);
}
