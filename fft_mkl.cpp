#include <fstream>
#include <fmt/core.h>
#include <chrono>
#include <mkl.h>

int main()
{
    std::ifstream in{"test_data"};
    size_t n; in >> n;
    const size_t buffer_size = 1024 * 1024;
    size_t count = 0, num = 0;
    std::vector<char> buffer; buffer.reserve(buffer_size + 100);
    MKL_Complex16* d_signal = reinterpret_cast<MKL_Complex16*>(mkl_malloc(n * sizeof(MKL_Complex16), 64));
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
            } else if(numHasValue) {
                d_signal[count].real = num;
                d_signal[count].imag = 0.0;
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
    //mkl_set_num_threads(mkl_get_max_threads());
    DFTI_DESCRIPTOR_HANDLE descriptor;
    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, n);
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiCommitDescriptor(descriptor);
    auto fftw_plan_end = std::chrono::steady_clock::now();
    fmt::println("Time taken to create FFT plan: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(fftw_plan_end - fftw_plan_start).count());
    
    auto fft_start = std::chrono::steady_clock::now();
    DftiComputeForward(descriptor, d_signal, d_signal);
    auto fft_end = std::chrono::steady_clock::now();
    fmt::println("Time taken to execute FFT: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(fft_end - fft_start).count());

    auto file_write_start = std::chrono::steady_clock::now();
    std::ofstream out{"mkl_output"};
    std::string output_buffer;
    output_buffer += fmt::format("{}\n", n);
    for (size_t count = 0; count < n; ++count) {
        output_buffer += fmt::format("{}{:+f}i\n", d_signal[count].real, d_signal[count].imag);
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
    DftiFreeDescriptor(&descriptor);
    mkl_free(d_signal);
}
