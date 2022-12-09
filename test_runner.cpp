#include <map>
#include <iostream>
#include <random>
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>
#include <easyvk.h>

using namespace std;
using namespace easyvk;

/** The test reads memory twice. */
const int readOutputs = 2;

/** MP-CO has nine possible behaviors. */
const int numBehaviors = 9;

/** Results */
typedef struct TestResults {
  int seq0;
  int seq1;
  int interleaved0;
  int interleaved1;
  int interleaved2;
  int interleaved3;
  int weak0;
  int weak1;
  int weak2;
} TestResults;

/** Returns the GPU to use for this test run. Users can specify the specific GPU to use
 *  with the 'gpuDeviceId' parameter. If gpuDeviceId is not included in the parameters or the specified
 *  device cannot be found, the first device is used.
 */
Device getDevice(Instance &instance, map<string, int> params) {
  int idx = 0;
  if (params.find("gpuDeviceId") != params.end()) {
    int j = 0;
    for (Device _device : instance.devices()) {
      if (_device.properties().deviceID == params["gpuDeviceId"]) {
        idx = j;
	break;
      }
      j++;
    }
  }
  Device device = instance.devices().at(idx);
  cout << "Using device " << device.properties().deviceName << "\n";
  return device;
}

/** Zeroes out the specified buffer. */
void clearMemory(Buffer &gpuMem, int size) {
  for (int i = 0; i < size; i++) {
    gpuMem.store(i, 0);
  }
}

/** Assigns shuffled workgroup ids, using the shufflePct to determine whether the ids should be shuffled this iteration. */
void setShuffledLocations(Buffer &shuffledLocations, int testingThreads) {
  vector<uint> locs;
  for (int i = 0; i < testingThreads; i++) {
    locs.push_back(i);
  }
  unsigned seed = chrono::system_clock::now().time_since_epoch().count();
  shuffle(locs.begin(), locs.end(), default_random_engine(seed));
  for (int i = 0; i < testingThreads; i++) {
      shuffledLocations.store(i, locs[i]);
  }
}

/** A test consists of N iterations of a shader and its corresponding result shader. */
void run(string &shader_file, map<string, int> params)
{
  // initialize settings
  auto instance = Instance(false);
  auto device = getDevice(instance, params);
  int workgroups = params["workgroups"];
  int workgroupSize = params["workgroupSize"];
  int testingThreads = workgroupSize * workgroups;
  TestResults finalResults;
  finalResults.seq0 = 0;
  finalResults.seq1 = 0;
  finalResults.interleaved0 = 0;
  finalResults.interleaved1 = 0;
  finalResults.interleaved2 = 0;
  finalResults.interleaved3 = 0;
  finalResults.weak0 = 0;
  finalResults.weak1 = 0;
  finalResults.weak2 = 0;

  // set up buffers
  auto testLocations = Buffer(device, testingThreads);
  auto shuffledLocations = Buffer(device, testingThreads);
  auto testResults = Buffer(device, numBehaviors);
  vector<Buffer> buffers = {testLocations, testResults, shuffledLocations};

  // run iterations
  for (int i = 0; i < params["iterations"]; i++) {
    auto program = Program(device, shader_file.c_str(), buffers);
    clearMemory(testLocations, testingThreads);
    clearMemory(testResults, numBehaviors);
    setShuffledLocations(shuffledLocations, testingThreads);
    program.setWorkgroups(workgroups);
    program.setWorkgroupSize(workgroupSize);
    program.prepare();
    program.run();
    finalResults.seq0 += testResults.load(0);
    finalResults.seq1 += testResults.load(1);
    finalResults.interleaved0 += testResults.load(2);
    finalResults.interleaved1 += testResults.load(3);
    finalResults.interleaved2 += testResults.load(4);
    finalResults.interleaved3 += testResults.load(5);
    finalResults.weak0 += testResults.load(6);
    finalResults.weak1 += testResults.load(7);
    finalResults.weak2 += testResults.load(8);
    program.teardown();
  }

  cout << "Results:\n";
  cout << "r0 == 0 && r1 == 0: " << finalResults.seq0 << "\n";
  cout << "r0 == 2 && r1 == 2: " << finalResults.seq1 << "\n";
  cout << "r0 == 1 && r1 == 1: " << finalResults.interleaved0 << "\n";
  cout << "r0 == 0 && r1 == 1: " << finalResults.interleaved1 << "\n";
  cout << "r0 == 0 && r1 == 2: " << finalResults.interleaved2 << "\n";
  cout << "r0 == 1 && r1 == 2: " << finalResults.interleaved3 << "\n";
  cout << "r0 == 1 && r1 == 0: " << finalResults.weak0 << "\n";
  cout << "r0 == 2 && r1 == 0: " << finalResults.weak1 << "\n";
  cout << "r0 == 2 && r1 == 0: " << finalResults.weak2 << "\n";
  cout << "\n";

  for (Buffer buffer : buffers) {
    buffer.teardown();
  }
  device.teardown();
  instance.teardown();
}

/** Reads a specified config file and stores the parameters in a map. Parameters should be of the form "key=value", one per line. */
map<string, int> read_config(string &config_file)
{
  map<string, int> m;
  ifstream in_file(config_file);
  string line;
  while (getline(in_file, line))
  {
    istringstream is_line(line);
    string key;
    if (getline(is_line, key, '='))
    {
      string value;
      if (getline(is_line, value))
      {
        m[key] = stoi(value);
      }
    }
  }
  return m;
}

void print_help()
{
  cout << "Usage: ./TestRunner paramFile\n";
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    print_help();
  }
  else
  {
    string configFile(argv[1]);
    map<string, int> params = read_config(configFile);
     for (const auto& [key, value] : params) {
        std::cout << key << " = " << value << "; ";
    }
    std::cout << "\n";
    string shaderFile("message-passing-coherency.spv");
    run(shaderFile, params);
  }
  return 0;
}
