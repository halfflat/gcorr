// Some initial pseudocode and thoughts from Adam
#include "fxkernel.h"
#include "math.h"
#include <stdio.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>

// FxKernel will operate on a single subband (dual pol), upper sideband, for the duration of one subintegration
// Suggest we fix to use 2 bit real data, in 2's complement?  i.e., assume that the data itself has headers stripped etc
//   - Ideally we would generate this by stripping out real data from e.g. VDIF files
// Will we use pthread parallelisation to do time division multiplexing like in DiFX, or use something else? Or simply run multiple, 
// completely independent instances of FxKernel (probably easier)?
// Suggest that we force the nchan to be a power of 2 to make the striding complex multiplication for phase rotations easy
// I'm also suggesting that we unpack directly to a complex array, to make the subsequent fringe rotation easier

static cf32 lut2bit[256][4];

void initLUT2bitReal () {
  static const float HiMag = 3.3359;  // Optimal value
  const float lut4level[4] = {-HiMag, -1.0, 1.0, HiMag};
  
  int l;
  for (int b = 0; b < 256; b++)	{
    for (int i = 0; i < 4; i++) {
      l = (b >> (2*i)) & 0x03;
      lut2bit[b][i].re = lut4level[l];
      lut2bit[b][i].im = 0;
    }
  }
}

FxKernel::FxKernel(int nant, int nchan, int nfft, int numbits, double lo, double bw)
  : numantennas(nant), numchannels(nchan), fftchannels(2*nchan), numffts(nfft), nbits(numbits), lofreq(lo), bandwidth(bw), sampletime(1.0/(2.0*bw))
{
  iscomplex = 0; // Allow for further generalisation later
  if (iscomplex)
  {
    cfact = 2;
    fftchannels /= 2;
    sampletime *= 2;
  }
  else
  {
    cfact = 1;
  }
  
  std::cout << "Subint time is " << sampletime*fftchannels*numffts*1000.0 << " msec" << std::endl;

  // Check for consistency and initialise lookup tables, if required.
  if (nbits==2) {
    if (fftchannels % 2) {
      std::cerr << "Error: FFT length must be divisible by 2 for 2 bit data. Aborting" << std::endl;  // 2 samples, 2 pols per byte
      exit(1);
    }
    initLUT2bitReal();
  } else if (nbits!=8) {
    std::cerr << "Error: Do not support " << nbits << " bits. Aborting" << std::endl;  //
    exit(1);
  }
  
  // Figure out the array stride size
  stridesize = (int)sqrt(numchannels);
  if(stridesize*stridesize != numchannels)
  {
    std::cerr << "Please choose a number of channels that is a square" << std::endl;
    exit(1);
  }
  substridesize = (2/cfact)*stridesize;

  // check if LO frequency has a fractional component
  fractionalLoFreq = false;
  if(lofreq - int(lofreq) > TINY)
  {
    fractionalLoFreq = true;
  }

  // allocate the unpacked array
  unpacked = new cf32**[nant];
  for(int i=0;i<nant;i++)
  {
    unpacked[i] = new cf32*[2];
    for(int j=0;j<2;j++)
    {
      unpacked[i][j] = vectorAlloc_cf32(fftchannels);
    }
  }

  //allocate the arrays for holding the fringe rotation vectors
  subtoff  = vectorAlloc_f64(substridesize);
  subtval  = vectorAlloc_f64(substridesize);
  subxoff  = vectorAlloc_f64(substridesize);
  subxval  = vectorAlloc_f64(substridesize);
  subphase = vectorAlloc_f64(substridesize);
  subarg   = vectorAlloc_f32(substridesize);
  subsin   = vectorAlloc_f32(substridesize);
  subcos   = vectorAlloc_f32(substridesize);
  steptoff  = vectorAlloc_f64(stridesize);
  steptval  = vectorAlloc_f64(stridesize);
  stepxoff  = vectorAlloc_f64(stridesize);
  stepxval  = vectorAlloc_f64(stridesize);
  stepphase = vectorAlloc_f64(stridesize);
  steparg   = vectorAlloc_f32(stridesize);
  stepsin   = vectorAlloc_f32(stridesize);
  stepcos   = vectorAlloc_f32(stridesize);
  stepcplx  = vectorAlloc_cf32(stridesize);
  complexrotator = vectorAlloc_cf32(fftchannels);

  // populate the fringe rotation arrays that can be pre-populated
  for(int i=0;i<substridesize;i++)
  {
    subxoff[i] = (double(i)/double(fftchannels));
    subtoff[i] = i*sampletime;
  }
  for(int i=0;i<stridesize;i++) 
  {
    stepxoff[i] = double(i*stridesize)/double(fftchannels);
    steptoff[i] = i*stridesize*sampletime;
  }

  // Allocate memory for FFT'ed data and initialised FFT
  int order = 0;
  while((fftchannels) >> order != 1)
  {
    order++;
  }
  
  channelised = new cf32**[nant];
  conjchannels = new cf32**[nant];
  for(int i=0;i<nant;i++)
  {
    channelised[i] = new cf32*[2];
    conjchannels[i] = new cf32*[2];
    for(int j=0;j<2;j++)
    {
      channelised[i][j] = vectorAlloc_cf32(fftchannels);
      conjchannels[i][j] = vectorAlloc_cf32(numchannels);
    }
  }
  
  // Get the size of, and initialise, the FFT
  int sizeFFTSpec, sizeFFTInitBuf, wbufsize;
  u8 *fftInitBuf, *fftSpecBuf;
  ippsFFTGetSize_C_32fc(order, vecFFT_NoReNorm, vecAlgHintFast, &sizeFFTSpec, &sizeFFTInitBuf, &wbufsize);
  fftSpecBuf = ippsMalloc_8u(sizeFFTSpec);
  fftInitBuf = ippsMalloc_8u(sizeFFTInitBuf);
  fftbuffer = ippsMalloc_8u(wbufsize);
  ippsFFTInit_C_32fc(&pFFTSpecC, order, vecFFT_NoReNorm, vecAlgHintFast, fftSpecBuf, fftInitBuf);
  if (fftInitBuf) ippFree(fftInitBuf);

  // Visibilities
  nbaselines = nant*(nant-1)/2;
  visibilities =  new cf32**[nbaselines];
  for(int i=0;i<nbaselines;i++)
  {
    visibilities[i] = new cf32*[4]; // 2 pols and crosspol
    for(int j=0; j<4; j++)
    {
      visibilities[i][j] = vectorAlloc_cf32(numchannels);
    }
  }
  
  // also the channel frequency arrays and the other fractional sample correction arrays
  subchannelfreqs = vectorAlloc_f32(stridesize);
  stepchannelfreqs = vectorAlloc_f32(stridesize);
  subfracsamparg = vectorAlloc_f32(stridesize);
  subfracsampsin = vectorAlloc_f32(stridesize);
  subfracsampcos = vectorAlloc_f32(stridesize);
  stepfracsamparg = vectorAlloc_f32(stridesize);
  stepfracsampsin = vectorAlloc_f32(stridesize);
  stepfracsampcos = vectorAlloc_f32(stridesize);
  stepfracsampcplx = vectorAlloc_cf32(stridesize);
  fracsamprotator = vectorAlloc_cf32(numchannels);

  // populate the channel frequency arrays
  for(int i=0;i<stridesize;i++)
  {
    subchannelfreqs[i] = (float)((TWO_PI*i*bandwidth)/(double)numchannels);
    stepchannelfreqs[i] = (float)((TWO_PI*i*stridesize*bandwidth)/(double)numchannels);
  }
}

FxKernel::~FxKernel()
{

  //de-allocate the internal arrays
  for(int i=0;i<numantennas;i++)
  {
    for(int j=0;j<2;j++)
    {
      vectorFree(unpacked[i][j]);
      vectorFree(channelised[i][j]);
    }
    delete [] unpacked[i];
    delete [] channelised[i];
  }
  delete [] unpacked;
  delete [] channelised;

  for(int i=0;i<nbaselines;i++)
  {
    for(int j=0;j<4;j++)
    {
      vectorFree(visibilities[i][j]);
    }
    delete [] visibilities[i];
  }
  delete [] visibilities;

  vectorFree(subtoff);
  vectorFree(subtval);
  vectorFree(subxoff);
  vectorFree(subxval);
  vectorFree(subphase);
  vectorFree(subarg);
  vectorFree(subsin);
  vectorFree(subcos);
  vectorFree(steptoff);
  vectorFree(steptval);
  vectorFree(stepxoff);
  vectorFree(stepxval);
  vectorFree(stepphase);
  vectorFree(steparg);
  vectorFree(stepsin);
  vectorFree(stepcos);
  vectorFree(stepcplx);
  vectorFree(complexrotator);
  vectorFree(subchannelfreqs);
  vectorFree(stepchannelfreqs);
  vectorFree(subfracsamparg);
  vectorFree(subfracsampsin);
  vectorFree(subfracsampcos);
  vectorFree(stepfracsamparg);
  vectorFree(stepfracsampsin);
  vectorFree(stepfracsampcos);
  vectorFree(stepfracsampcplx);
  vectorFree(fracsamprotator);
}

void FxKernel::setInputData(u8 ** idata)
{
  inputdata = idata;
}

void FxKernel::setDelays(double ** d)
{
  delays = d;
}

void FxKernel::process()
{
  // delay variables
  double meandelay; //mean delay in the middle of the FFT for a given antenna
  double fractionaldelay; // fractional component of delay to be correction after channelisation
  double delaya, delayb; // coefficients a and b for the interpolation across a given FFT
  int sampledelay; //integer number of samples delay
  int offset; //offset into the packed data vector

  // Zero visibilities
  for (int i=0;i<nbaselines; i++)
  {
    for (int j=0; j<4; j++)
    {
      vectorZero_cf32(visibilities[i][j], numchannels);
    }
  }
  
  // for(number of FFTs)... (parallelised via pthreads?)
  for(int i=0;i<numffts;i++)
  {
    // do station-based processing for each antenna in turn
    for(int j=0;j<numantennas;j++)
    {
      // FIXME: as written, having taken out the bulk delay provided here (referencing the delay
      // to the start of the data) would mean that the fringe rotation wouldn't be right (it 
      // wouldn't line up across subintegrations) but that isn't relevant from a benchmarking 
      // point of view.  However, at some point it would probably be better to provide a full, correct
      // delay to each station, and to also provide the value by which each station datastream had
      // already been offset (and hence do this properly).
      //
      // FIXME: In DiFX we maximise cache efficiency by doing multiple FFTs, then doing multiple cross-multiplcations.
      // We don't have that here, and it can make a difference of 10s of percent, so we should take that
      // into account when making any comparisons to the GPU.  We can always quantify the effect in DiFX
      // by setting numBufferedFFTs to 1 and xmacstridelength to be numchannels, and looking at the reduction
      // in performance.

      // unpack
      getStationDelay(j, i, meandelay, delaya, delayb);
      double delayinsamples = meandelay / sampletime;
      sampledelay = int(delayinsamples + 0.5);

      fractionaldelay = (delayinsamples - sampledelay)*sampletime;  // seconds
      offset = i*fftchannels - sampledelay;
      if(offset < 0) 
      {
        if(offset == -1) // can happen due to difference in geometric delay between start of first FFT and middle of first FFT
        {
          offset += 1;
          fractionaldelay += sampletime;
        }
        else // must have made a mistake with the delay polynomial
        {
          std::cerr << "Offset is " << offset << " samples, this should never happen.  Aborting" << std::endl;
          exit(1);
        }
      }
      unpack(inputdata[j], unpacked[j], offset);
  
      // fringe rotate - after this function, each unpacked array has been fringe-rotated in-place
      fringerotate(unpacked[j], delaya, delayb);

      // Channelise
      dofft(unpacked[j], channelised[j]);

      // If original data was real voltages, required channels will fill n/2+1 of the n channels, so move in-place
    
      // Fractional sample correct
      fracSampleCorrect(channelised[j], fractionaldelay);

      // Calculate complex conjugate once, for efficency
      conjChannels(channelised[j], conjchannels[j]);
    }

    // then do the baseline based processing   (CJP: What about autos)
    int b = 0; // Baseline counter
    for(int j=0;j<numantennas-1;j++)
    {
      for(int k=j+1;k<numantennas;k++)
      {
	for(int l=0;l<2;l++)
	{
	  for(int m=0;m<2;m++)
	  {
	    // cross multiply + accumulate
	    vectorAddProduct_cf32(channelised[j][l], conjchannels[k][m], visibilities[b][2*l + m], numchannels);
	  }
	}
	b++;
      }
    }
  }

  // Normalise
  cf32 norm;
  norm.re = numffts;
  norm.im = 0;
  for (int i=0; i<nbaselines; i++) {
    for (int j=0; j<4; j++) {
      vectorDivC_cf32_I(norm, visibilities[i][j], numchannels);
    }
  }
}

void FxKernel::saveVisibilities(const char * outfile) {
  f32 ***amp, ***phase;

  std::ofstream fvis(outfile, std::ios::binary);
  
  for (int i=0;i<nbaselines;i++)  {
    for (int j=0; j<4; j++) {
      fvis.write(reinterpret_cast<char *>(visibilities[i][j]), numchannels * sizeof(cf32));
    }
  }
}

void FxKernel::accumulate(cf32 *** odata)
{
  for (int i=0; i<nbaselines; i++)
  {
    for (int j=0; j<4; j++)
    {
      vectorAdd_cf32_I(visibilities[i][j], odata[i][j], numchannels);
    }
  }
}

void FxKernel::getStationDelay(int antenna, int fftindex, double & meandelay, double & a, double & b)
{
  double * interpolator = delays[antenna];
  double delta = fftchannels * sampletime;
  double delta2 = delta * delta;
  double d0, d1, d2;
  int integerdelay;

  // calculate values at the beginning, middle, and end of this FFT
  d0 = interpolator[0]*delta*delta + interpolator[1]*delta + interpolator[2];
  d1 = interpolator[0]*(fftindex+0.5)*(fftindex+0.5)*delta*delta + interpolator[1]*(fftindex+0.5)*delta + interpolator[2];
  d2 = interpolator[0]*(fftindex+1.0)*(fftindex+1.0)*delta*delta + interpolator[1]*(fftindex+1.0)*delta + interpolator[2];

  // use these to calculate a linear interpolator across the FFT, as well as a mean value
  a = d2-d0;
  b = d0 + (d1 - (a*0.5 + d0))/3.0;
  meandelay = a*0.5 + b;
}

#if 0
// 2 bit, 2 channel unpacker 
void unpackReal2bit(u8 * inputdata, cf32 ** unpacked, int offset, int nsamp) {
  // inputdata     array of tightly packed 2bit samples, 4 samples (2 times, 2 channels) per byte. Channels are interleaved
  // unpacked      data as 2 complex float arrays
  // offset        offset into array in SAMPLES
  // nsamp         number of samples to unpack
  cf32 *fp;

  int o = 0;
  u8 *byte = &inputdata[offset/2];
  if (offset%2) { 
    fp = lut2bit[*byte]; 
    unpacked[0][o] = fp[2];
    unpacked[1][o] = fp[3];
    o++;
    byte++;
    nsamp--;
  }
  
  for (; o<nsamp-1; o++) { // 2 time samples/byte
    fp = lut2bit[*byte];  // pointer to vector of 4 complex floats
    byte++;               // move pointer to next byte for next iteration of loop
    unpacked[0][o] = fp[0];
    unpacked[1][o] = fp[1];
    o++;
    unpacked[0][o] = fp[2];
    unpacked[1][o] = fp[3];
  }

  if (nsamp%2) {
    fp = lut2bit[*byte]; 
    unpacked[0][o] = fp[0];
    unpacked[1][o] = fp[1];
  }
}
#endif

// 2 bit, 2 channel unpacker 
void unpackReal2bit(u8 * inputdata, cf32 ** unpacked, int offset, int nsamp) {
  // inputdata     array of tightly packed 2bit samples, 4 samples (2 times, 2 channels) per byte. Channels are interleaved
  // unpacked      data as 2 complex float arrays
  // offset        offset into array in SAMPLES
  // nsamp         number of samples to unpack
  cf32 *fp;

  int o = 0;
  int i = offset % 4;
  u8 *byte = &inputdata[offset/4];
  
  for (; o<nsamp; o++) { // 2 time samples/byte
    fp = lut2bit[*byte];  // pointer to vector of 4 complex floats
    for (; i < 4 && o < nsamp; i++) {
      unpacked[0][o] = fp[i];
      unpacked[1][o] = fp[i];
      o++;
      i++;
    }
    byte++;               // move pointer to next byte for next iteration of loop
    i = 0;
  }
}

void FxKernel::unpack(u8 * inputdata, cf32 ** unpacked, int offset)
{
  if (nbits==2) {
    unpackReal2bit(inputdata, unpacked, offset, fftchannels);
  } else {
    std::cerr << "Unsupported number of bits!!!" << std::endl;
  }
}

void FxKernel::fringerotate(cf32 ** unpacked, f64 a, f64 b)
{
  int integerdelay;
  int status;

  // subtract off any integer delay present
  integerdelay = static_cast<int>(b/sampletime) * sampletime;
  b -= integerdelay;

  // Fill in the delay values, using a and b and the precomputeed offsets
  status = vectorMulC_f64(subxoff, a, subxval, substridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in linearinterpolate, subval multiplication\n");
  status = vectorMulC_f64(stepxoff, a, stepxval, stridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in linearinterpolate, stepval multiplication\n");
  status = vectorAddC_f64_I(b, subxval, substridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in linearinterpolate, subval addition!!!\n");

  // Turn delay into turns of phase by multiplying by the lo
  status = vectorMulC_f64(subxval, lofreq, subphase, substridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in linearinterpolate lofreq sub multiplication!!!\n");
  status = vectorMulC_f64(stepxval, lofreq, stepphase, stridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in linearinterpolate lofreq step multiplication!!!\n");
  if(fractionalLoFreq) 
  {
    status = vectorAddC_f64_I((lofreq-int(lofreq))*double(integerdelay), subphase, substridesize);
    if(status != vecNoErr)
      fprintf(stderr, "Error in linearinterpolate lofreq non-integer freq addition!!!\n");
  }

  // Convert turns of phase into radians and bound into [0,2pi), then take sin/cos and assemble rotator vector
  for(int i=0;i<substridesize;i++) 
  {
    subarg[i] = -TWO_PI*(subphase[i] - int(subphase[i]));
  }
  for(int i=0;i<stridesize;i++)
  {
    steparg[i] = -TWO_PI*(stepphase[i] - int(stepphase[i]));
  }
  status = vectorSinCos_f32(subarg, subsin, subcos, substridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in sin/cos of sub rotate argument!!!\n");
  status = vectorSinCos_f32(steparg, stepsin, stepcos, stridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error in sin/cos of step rotate argument!!!\n");
  status = vectorRealToComplex_f32(subcos, subsin, complexrotator, substridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error assembling sub into complex!!!\n");
  status = vectorRealToComplex_f32(stepcos, stepsin, stepcplx, stridesize);
  if(status != vecNoErr)
    fprintf(stderr, "Error assembling step into complex!!!\n");
  for(int i=1;i<stridesize;i++) 
  {
    status = vectorMulC_cf32(complexrotator, stepcplx[i], &complexrotator[i*substridesize], substridesize);
    if(status != vecNoErr)
      fprintf(stderr, "Error doing the time-saving complex multiplication!!!\n");
  }

  // Actually apply the fringe rotation to each polarisation in turn
  for (int i=0;i<2;i++)
  {
    status = vectorMul_cf32_I(complexrotator, unpacked[i], fftchannels);
    if (status != vecNoErr)
      std::cerr << "Error in complex fringe rotation" << std::endl;
  }
}

void FxKernel::dofft(cf32 ** unpacked, cf32 ** channelised) {
  // Do a single FFT on the 2 pols for a single antenna
  vecStatus status;
  
  for (int i=0; i<2; i++) {
    status = vectorFFT_CtoC_cf32(unpacked[i], channelised[i], pFFTSpecC, fftbuffer);
    if(status != vecNoErr) {
      std::cerr << "Error calling FFT" << std::endl;
      exit(1);
    }
  }
}

void FxKernel::conjChannels(cf32 ** channelised, cf32 ** conjchannels) {
  // To avoid calculating this multiple times, generate the complex conjugate of the channelised data 
  vecStatus status;
  
  for (int i=0; i<2; i++) {
    status = vectorConj_cf32(channelised[i], conjchannels[i], numchannels); // Assumes USB and throws away 1 channel for real data
    if(status != vecNoErr) {
      std::cerr << "Error calling vectorConj" << std::endl;
      exit(1);
    }
  }
}

void FxKernel::fracSampleCorrect(cf32 ** channelised, f64 fracdelay)
{
  int status;

  // Create an array of phases for the fractional sample correction
  status = vectorMulC_f32(subchannelfreqs, fracdelay, subfracsamparg, stridesize);
  if(status != vecNoErr)
    std::cerr << "Error in frac sample correction, arg generation (sub)!!!" << status << std::endl;
  status = vectorMulC_f32(stepchannelfreqs, fracdelay, stepfracsamparg, stridesize);
  if(status != vecNoErr)
    std::cerr << "Error in frac sample correction, arg generation (step)!!!" << status << std::endl;

  //create the fractional sample correction array
  status = vectorSinCos_f32(subfracsamparg, subfracsampsin, subfracsampcos, stridesize);
  if(status != vecNoErr)
    std::cerr << "Error in frac sample correction, sin/cos (sub)!!!" << status << std::endl;
  status = vectorSinCos_f32(stepfracsamparg, stepfracsampsin, stepfracsampcos, stridesize);
  if(status != vecNoErr)
    std::cerr << "Error in frac sample correction, sin/cos (sub)!!!" << status << std::endl;
  status = vectorRealToComplex_f32(subfracsampcos, subfracsampsin, fracsamprotator, stridesize);
  if(status != vecNoErr)
    std::cerr << "Error in frac sample correction, real to complex (sub)!!!" << status << std::endl;
  status = vectorRealToComplex_f32(stepfracsampcos, stepfracsampsin, stepfracsampcplx, stridesize);
  if(status != vecNoErr)
    std::cerr << "Error in frac sample correction, real to complex (step)!!!" << status << std::endl;
  for(int i=1;i<stridesize;i++)
  {
    status = vectorMulC_cf32(fracsamprotator, stepfracsampcplx[i], &(fracsamprotator[i*stridesize]), stridesize);
    if(status != vecNoErr)
      std::cerr << "Error doing the time-saving complex multiplication in frac sample correction!!!" << std::endl;
  }

  // Apply the fractional sample correction to each polarisation in turn
  for(int i=0;i<2;i++)
  {
    status = vectorMul_cf32_I(fracsamprotator, channelised[i], numchannels);
    if(status != vecNoErr)
      std::cerr << "Error in application of frac sample correction!!!" << status << std::endl;
  }
}
