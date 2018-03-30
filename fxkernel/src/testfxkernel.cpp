#include "fxkernel.h"

int main(int argc, char *argv[])
{
  // variables for the test
  char ** inputdata;
  double ** delays;
  int numchannels, numantennas, numffts;
  double lo, bandwidth;

  // load up the test input data from somewhere
  
  // Load up the delays from somewhere

  // Set the inputs we'll use as a test
  // Current values would give a subint of 100ms, which is fairly reasonable
  numchannels = 1024;
  numantennas = 6;
  numffts = 3125;
  lo = 1650000000.0;
  bandwidth = 32000000.0;

  // create the FxKernel
  // We could also create multiple FxKernels to test parallelisation in a simple/lazy way
  FxKernel fxkernel = FxKernel(numchannels, numantennas, numffts, lo, bandwidth);

  // Set the input data and the delays
  fxkernel.setInputData(inputdata);
  fxkernel.setDelays(delays);

  // Checkpoint for timing
  
  // Run the processing
  fxkernel.process();

  // Calculate the elapsed time
  
}
