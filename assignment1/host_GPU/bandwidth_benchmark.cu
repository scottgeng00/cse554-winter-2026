#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>

const int NUM_ITERATIONS = 100;

int main() {
    // we write results to a csv which we will use to plot.
    std::ofstream csv("bandwidth_results.csv");
    csv << "power_of_2,bytes,h2d_bandwidth_GBps,d2h_bandwidth_GBps,h2d_bandwidth_GBps_pinned,d2h_bandwidth_GBps_pinned\n";
        
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int power = 0; power <= 20; power++) {
        size_t bytes = 1ULL << power;
        
        void *h_data = malloc(bytes);
        void *d_data;
        cudaMalloc(&d_data, bytes);
        memset(h_data, 0xAB, bytes);
        
        // Measure Host-to-Device
        cudaEventRecord(start);
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float h2d_ms;
        cudaEventElapsedTime(&h2d_ms, start, stop);
        // bytes * NUM_ITERATIONS is the total number of bytes transferred
        // h2d_ms * 1e-3 to get time in seconds
        // 1024^3 to convert bytes to GB
        double h2d_bw = static_cast<double>(bytes * NUM_ITERATIONS) / (h2d_ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);
        
        // Measure Device-to-Host
        cudaEventRecord(start);
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float d2h_ms;
        cudaEventElapsedTime(&d2h_ms, start, stop);
        double d2h_bw = static_cast<double>(bytes * NUM_ITERATIONS) / (d2h_ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);
        free(h_data);
        cudaFree(d_data);

        // pinned memory
        void *h_data_pinned;
        void *d_data_pinned;
        cudaMallocHost(&h_data_pinned, bytes);
        cudaMalloc(&d_data_pinned, bytes);
        memset(h_data_pinned, 0xAB, bytes);
        
        cudaEventRecord(start);
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            cudaMemcpy(d_data_pinned, h_data_pinned, bytes, cudaMemcpyHostToDevice);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float h2d_ms_pinned;
        cudaEventElapsedTime(&h2d_ms_pinned, start, stop);
        double h2d_bw_pinned = static_cast<double>(bytes * NUM_ITERATIONS) / (h2d_ms_pinned * 1e-3) / (1024.0 * 1024.0 * 1024.0);
        
        cudaEventRecord(start);
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            cudaMemcpy(h_data_pinned, d_data_pinned, bytes, cudaMemcpyDeviceToHost);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float d2h_ms_pinned;
        cudaEventElapsedTime(&d2h_ms_pinned, start, stop);
        double d2h_bw_pinned = static_cast<double>(bytes * NUM_ITERATIONS) / (d2h_ms_pinned * 1e-3) / (1024.0 * 1024.0 * 1024.0);
        
        csv << power << "," << bytes << "," << h2d_bw << "," << d2h_bw << "," << h2d_bw_pinned << "," << d2h_bw_pinned << "\n";
        
        cudaFreeHost(h_data_pinned);
        cudaFree(d_data_pinned);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Results saved to bandwidth_results.csv" << std::endl;
    
    return 0;
}
