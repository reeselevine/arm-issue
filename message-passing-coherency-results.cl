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

__kernel void litmus_test (
  __global atomic_uint* read_results,
  __global TestResults* test_results) {
  uint id_0 = get_global_id(0);
  uint r0 = atomic_load(&read_results[id_0 * 2]);
  uint r1 = atomic_load(&read_results[id_0 * 2 + 1]);
  if ((r0 == 0 && r1 == 0)) {
    atomic_fetch_add(&test_results->seq0, 1);
  } else if ((r0 == 2 && r1 == 2)) {
    atomic_fetch_add(&test_results->seq1, 1);
  } else if ((r0 == 1 && r1 == 1)) {
    atomic_fetch_add(&test_results->interleaved0, 1);
  } else if ((r0 == 0 && r1 == 1)) {
    atomic_fetch_add(&test_results->interleaved1, 1);
  } else if ((r0 == 0 && r1 == 2)) {
    atomic_fetch_add(&test_results->interleaved2, 1);
  } else if ((r0 == 1 && r1 == 2)) {
    atomic_fetch_add(&test_results->interleaved3, 1);
  } else if ((r0 == 1 && r1 == 0)) {
    atomic_fetch_add(&test_results->weak0, 1);
  } else if ((r0 == 2 && r1 == 0)) {
    atomic_fetch_add(&test_results->weak1, 1);
  } else if ((r0 == 2 && r1 == 1)) {
    atomic_fetch_add(&test_results->weak2, 1);
  }
}
    
