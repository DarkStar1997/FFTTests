#include <fstream>
#include <fmt/core.h>
#include <chrono>
#include <fftw3.h>

int main()
{
    std::ifstream in{"test_data"};
    size_t n; in >> n;
    const size_t buffer_size = 1024 * 1024;
    size_t count = 0, num = 0;
    std::vector<char> buffer; buffer.reserve(buffer_size + 100);
    fftw_complex* d_signal = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * n));
    fmt::print("Reading {} integers\n", n);

    auto file_read_start = std::chrono::steady_clock::now();
    while(count < n) {
        in.read(buffer.data(), buffer_size);
        size_t len = in.gcount();
        if(len == 0)
            break;

        bool numHasValue = false;
        for(size_t i = 0; i < len; i++) {
            const char &ch = buffer[i];
            if(ch >= '0' && ch <= '9') {
                num = num * 10 + ch - '0';
                numHasValue = true;
            }
            else if(numHasValue) {
                d_signal[count][0] = num;
                d_signal[count][1] = 0.0;
                num = 0;
                count++;
                numHasValue = false;
            }
        }
    }
    
    buffer.clear(); buffer.shrink_to_fit();
    in.close();
    
    auto file_read_end = std::chrono::steady_clock::now();
    fmt::println("Time taken to read from file: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(file_read_end - file_read_start).count());
    
    auto fftw_plan_start = std::chrono::steady_clock::now();
    fftw_plan plan = fftw_plan_dft_1d(n, d_signal, d_signal, FFTW_FORWARD, FFTW_ESTIMATE);
    auto fftw_plan_end = std::chrono::steady_clock::now();
    fmt::println("Time taken to create FFT plan: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(fftw_plan_end - fftw_plan_start).count());
    
    auto fft_start = std::chrono::steady_clock::now();
    fftw_execute(plan);
    auto fft_end = std::chrono::steady_clock::now();
    fmt::println("Time taken to execute FFT: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(fft_end - fft_start).count());

    auto file_write_start = std::chrono::steady_clock::now();
    std::ofstream out{"cpu_output"};
    std::string output_buffer;
    output_buffer += fmt::format("{}\n", n);
    for (size_t count = 0; count < n; ++count) {
        output_buffer += fmt::format("{}{:+f}i\n", d_signal[count][0], d_signal[count][1]);
        if(output_buffer.length() >= buffer_size) {
            out.write(output_buffer.data(), output_buffer.length());
            output_buffer.clear();
        }
    }
    if(output_buffer.length() > 0) {
        out.write(output_buffer.data(), output_buffer.length());
    }
    auto file_write_end = std::chrono::steady_clock::now();
    out.close();
    fmt::println("Time taken to write to file: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(file_write_end - file_write_start).count());
    fftw_destroy_plan(plan);
    fftw_free(d_signal);
}
