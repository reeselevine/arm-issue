typedef struct TestResults {
  atomic_uint seq0;
  atomic_uint seq1;
  atomic_uint interleaved0;
  atomic_uint interleaved1;
  atomic_uint interleaved2;
  atomic_uint interleaved3;
  atomic_uint weak0;
  atomic_uint weak1;
  atomic_uint weak2;
} TestResults;

// Calling this function seems sufficient to avoid x_0/y_0 and x_1/y_1 from being
// coalesced, and allows the load reordering to occur.
static uint get_new_id(uint id) {
    return id;
}

__kernel void litmus_test (
  __global atomic_uint* test_locations,
  __global TestResults* test_results,
  __global uint* shuffled_locations) {
    uint id_0 = get_group_id(0) * get_local_size(0) + get_local_id(0);
    uint id_1 = shuffled_locations[id_0];
    uint x_0 = id_0;
    uint y_0 = get_new_id(id_0);
    uint x_1 = id_1;
    uint y_1 = get_new_id(id_1);
    atomic_store_explicit(&test_locations[x_0], 1, memory_order_relaxed);
    atomic_store_explicit(&test_locations[y_0], 2, memory_order_relaxed);
    uint r0 = atomic_load_explicit(&test_locations[y_1], memory_order_relaxed);
    uint r1 = atomic_load_explicit(&test_locations[x_1], memory_order_relaxed);
    if (r0 == 0 && r1 == 0) {
      atomic_fetch_add(&test_results->seq0, 1);
    } else if (r0 == 2 && r1 == 2) {
      atomic_fetch_add(&test_results->seq1, 1);
    } else if (r0 == 1 && r1 == 1) {
      atomic_fetch_add(&test_results->interleaved0, 1);
    } else if (r0 == 0 && r1 == 1) {
      atomic_fetch_add(&test_results->interleaved1, 1);
    } else if (r0 == 0 && r1 == 2) {
      atomic_fetch_add(&test_results->interleaved2, 1);
    } else if (r0 == 1 && r1 == 2) {
      atomic_fetch_add(&test_results->interleaved3, 1);
    } else if (r0 == 1 && r1 == 0) {
      atomic_fetch_add(&test_results->weak0, 1);
    } else if (r0 == 2 && r1 == 0) {
      atomic_fetch_add(&test_results->weak1, 1);
    } else if (r0 == 2 && r1 == 1) {
      atomic_fetch_add(&test_results->weak2, 1);
    }
}
