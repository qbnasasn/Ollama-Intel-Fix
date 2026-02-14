#include <sycl/sycl.hpp>

// Snippet to replace queue creation in ggml-sycl.cpp
// Location: ggml_backend_sycl_buffer_type() or similar init function

sycl::queue create_xmx_optimized_queue(int device_id) {
    auto device = sycl::device::get_devices()[device_id];
    
    // PROPERTY 1: Immediate Command Lists
    // drastically reduces submission latency by bypassing the scheduler
    auto prop_list = sycl::property_list{
        sycl::ext::intel::property::queue::immediate_command_list{},
        sycl::property::queue::in_order{} 
    };

    // PROPERTY 2: Priority Boost
    // Can be set via Context or env var, but queue priority hints help
    // sycl::ext::oneapi::property::queue::priority_high{} 

    sycl::queue q(device, prop_list);
    
    return q;
}
