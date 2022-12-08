static uint get_new_id(uint id) {
    return id;
}

__kernel void litmus_test (
  __global atomic_uint* test_locations,
  __global atomic_uint* read_results,
  __global uint* shuffled_locations) {
    uint id_0 = get_group_id(0) * get_local_size(0) + get_local_id(0);
    uint id_1 = shuffled_locations[id_0];
    uint x_0 = id_0;
    uint y_0 = get_new_id(id_0);
    uint x_1 = id_1;
    uint y_1 = get_new_id(id_0);
    atomic_store_explicit(&test_locations[x_0], 1, memory_order_relaxed);
    atomic_store_explicit(&test_locations[y_0], 2, memory_order_relaxed);
    uint r0 = atomic_load_explicit(&test_locations[y_1], memory_order_relaxed);
    uint r1 = atomic_load_explicit(&test_locations[x_1], memory_order_relaxed);
    atomic_store(&read_results[id_1 * 2], r0);
    atomic_store(&read_results[id_1 * 2 + 1], r1);
}
