#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <iostream>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

// Target: Intel Arc B580 (Battlemage)
// XMX Configuration for F16
#define TM 8
#define TN 16
#define TK 16

// Helper to create multi_ptr from raw pointer for USM
template <typename T>
auto make_global_ptr(T* ptr) {
    return sycl::address_space_cast<sycl::access::address_space::global_space, 
                                    sycl::access::decorated::no>(ptr);
}

// Simple XMX GEMM Kernel for F16 (Half)
// C = alpha * (A * B) + beta * C
template <typename T>
void xmx_gemm_kernel(queue &q, 
                     const T *A, const T *B, T *C, 
                     int M, int N, int K, 
                     T alpha, T beta) {
    
    range<2> global_size(M / TM, N / TN);
    range<2> local_size(1, 16); 

    q.submit([&](handler &h) {
        h.parallel_for(nd_range<2>(global_size, local_size), [=](nd_item<2> item) {
            auto sg = item.get_sub_group();
            
            int m_idx = item.get_global_id(0) * TM;
            int n_idx = item.get_global_id(1) * TN;

            joint_matrix<sub_group, T, use::a, TM, TK, layout::row_major> tA;
            joint_matrix<sub_group, T, use::b, TK, TN, layout::row_major> tB; // Using row_major for simplicity
            joint_matrix<sub_group, T, use::accumulator, TM, TN> tC;

            joint_matrix_fill(sg, tC, T(0.0f));

            for (int k = 0; k < K; k += TK) {
                // Cast to global multi_ptr for joint_matrix_load
                auto ptrA = make_global_ptr(A + m_idx * K + k);
                auto ptrB = make_global_ptr(B + k * N + n_idx);
                
                joint_matrix_load(sg, tA, ptrA, K);
                joint_matrix_load(sg, tB, ptrB, N);

                joint_matrix_mad(sg, tC, tA, tB, tC);
            }

            // Store Result
            auto ptrC = make_global_ptr(C + m_idx * N + n_idx);
            joint_matrix_store(sg, tC, ptrC, N, layout::row_major);
        });
    });
}

// Helper to launch the kernel
void ggml_sycl_xmx_gemm(queue &q, const void *vx, const void *vy, void *vz, 
                        int m, int n, int k) {
    const half *x = (const half *)vx;
    const half *y = (const half *)vy;
    half *z = (half *)vz;
    
    // Check alignment/size constraints if necessary
    // M must be multiple of TM, N of TN, K of TK
    // For now we assume padded or handle edge cases (omitted for PoC)
    
    xmx_gemm_kernel(q, x, y, z, m, n, k, (half)1.0f, (half)0.0f);
}

// Queue Creation Logic (Merged from patch file)
// Location: ggml_backend_sycl_buffer_type() or similar init function

sycl::queue create_xmx_optimized_queue(int device_id) {
    // Get device via dpct manager or directly
    // Since we are inside ggml-sycl.cpp context, use standard sycl
    auto devices = sycl::device::get_devices();
    if(device_id >= devices.size()) device_id = 0; 
    auto device = devices[device_id];
    
    auto prop_list = sycl::property_list{
        sycl::ext::intel::property::queue::immediate_command_list{},
        sycl::property::queue::in_order{} 
    };

    sycl::queue q(device, prop_list);
    return q;
}
