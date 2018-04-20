#include <cstddef>
#include <vector>
#include <iostream>
#include <random>

#include <cuComplex.h>

#include "gtest.h"

#include "gpu_array.h"
#include "gxkernel.h"

using std::size_t;

template <typename Wrapped>
double run_kernel(int repeat_count, Wrapped fn) {
    cudaEvent_t ev[2];
    cudaEventCreate(&ev[0]);
    cudaEventCreate(&ev[1]);

    cudaEventRecord(ev[0]);
    for (int i = 0; i<repeat_count; ++i) fn();
    cudaEventRecord(ev[1]);
    cudaEventSynchronize(ev[1]);

    float ms = 0;
    cudaEventElapsedTime(&ms, ev[0], ev[1]);
    cudaEventDestroy(ev[0]);
    cudaEventDestroy(ev[1]);

    return ms/1000.0/(double)repeat_count;
}

inline int nblocks(int n, int width) {
    return n? 1+(n-1)/width: 0;
}

std::vector<float2> run_CrossCorrAccumHoriz(const std::vector<float2>& data, int nant, int nfft, int nchan, int fftwidth) {
    constexpr int npol = 2;

    int block_width = 128;
    dim3 ccblock(nblocks(nchan, block_width), nant-1, nant-1);

    size_t result_sz = nant*(nant-1)/2*npol*npol*nchan;

    gpu_array<float2> gpu_data(data);
    gpu_array<float2> gpu_result(result_sz);

    CrossCorrAccumHoriz<<<ccblock, block_width>>>(gpu_result.data(), gpu_data.data(), nant, nfft, nchan, fftwidth);
    return gpu_result;
}

double time_CrossCorrAccumHoriz(int repeat_count, const float2* gpu_data, int nant, int nfft, int nchan, int fftwidth) {
    constexpr int npol = 2;

    int block_width = 128;
    dim3 ccblock(nblocks(nchan, block_width), nant-1, nant-1);

    size_t result_sz = nant*(nant-1)/2*npol*npol*nchan;
    gpu_array<float2> gpu_result(result_sz);

    return run_kernel(repeat_count,
	[&]() {
	    CrossCorrAccumHoriz<<<ccblock, block_width>>>(gpu_result.data(), gpu_data, nant, nfft, nchan, fftwidth);
	});
}

std::vector<float2> run_CCAH2(const std::vector<float2>& data, int nant, int nfft, int nchan, int fftwidth) {
    constexpr int npol = 2;
    int nantxp = nant*npol;

    int block_width = 128;
    dim3 ccblock(nblocks(nchan, block_width), nantxp-1, nantxp-1);

    size_t result_sz = nant*(nant-1)/2*npol*npol*nchan;

    gpu_array<float2> gpu_data(data);
    gpu_array<float2> gpu_result(result_sz);

    CCAH2<<<ccblock, block_width>>>(gpu_result.data(), gpu_data.data(), nant, nfft, nchan, fftwidth);
    return gpu_result;
}

double time_CCAH2(int repeat_count, const float2* gpu_data, int nant, int nfft, int nchan, int fftwidth) {
    constexpr int npol = 2;
    int nantxp = nant*npol;

    int block_width = 128;
    dim3 ccblock(nblocks(nchan, block_width), nantxp-1, nantxp-1);

    size_t result_sz = nant*(nant-1)/2*npol*npol*nchan;
    gpu_array<float2> gpu_result(result_sz);

    return run_kernel(repeat_count,
	[&]() {
	    CCAH2<<<ccblock, block_width>>>(gpu_result.data(), gpu_data, nant, nfft, nchan, fftwidth);
	});
}

// This is stupidly slow:
#if 0
__global__ void CCAH3(cuComplex *accum, const cuComplex *ants, int nant, int nfft, int nchan, int fftwidth) {
    extern __shared__ float2 h[];

    int t = threadIdx.x+blockIdx.x*blockDim.x;
    if (t>=nchan) return;
    int block_width = blockDim.x;

    // blockIdx.y: index of first vector (2*antennaindex+polindex)

    int s = nfft*fftwidth;

    int i = blockIdx.y;
    int j0 = 2*(i/2+1);

    int ai = i/2;
    int b = 4*(ai*nant-ai*(ai+1)/2) + 2*(i-2*ai);

    const float2* iv = ants+i*s+t;
    const float2* jv = ants+j0*s+t;

    float2 u = iv[0];
    int hoff = threadIdx.x;
    for (int j = j0; j<2*nant; ++j) {
	float2 v = jv[0];
	float2 z;
	z.x = u.x*v.x + u.y*v.y;
	z.y = u.y*v.x - u.x*v.y;
	h[hoff] = z;

	jv += s;
	hoff += block_width;
    }

    for (int k = fftwidth; k<s; k += fftwidth) {
	u = iv[k];
	jv = ants+j0*s+t;

	int hoff = threadIdx.x;
	for (int j = j0; j<2*nant; ++j) {
	    float2 v = jv[k];
	    float2 z;
	    z.x = u.x*v.x + u.y*v.y;
	    z.y = u.y*v.x - u.x*v.y;
	    h[hoff].x += z.x;
	    h[hoff].y += z.y;

	    jv += s;
	    hoff += block_width;
	}
    }

    float oonfft = 1.f/nfft;

    hoff = threadIdx.x;
    for (int j = j0; j<2*nant; ++j) {
	float2 a = h[hoff];
	a.x *= oonfft;
	a.y *= oonfft;

	int dj = j-j0;
	int aj = 2*(dj/2);
	accum[(b+aj*2+(dj-aj))*nchan+t] = a;
	hoff += block_width;
    }
}
#elif 1
constexpr int ccah3_cwidth = 4096;
// Plan:
//  1.  Use shared mem for vector i cache with blockDim.x == ccah3_width.'
//      split vector horizontally in griDim.x blocks, gridDim.x*blockDim.x>=nchan*nfft.
//      For ease of impl, presume nchan|ccah3_width.
//  2.  Each block loads k nchan-blocks for vector i into cache.
//  3.  Horizontal accumulation is performed serially across vectors j; either pre-zero
//      accumulator or run left-side (offset 0) kernel first.

template <bool initial>
__global__ void CCAH3(cuComplex *accum, const cuComplex *ants, int nant, int nfft, int nchan, int fftwidth) {
    extern __shared__ float2 h[ccah3_cwidth];

    assert(ccah_cwidth%nchan==0);
    assert(
    int t = threadIdx.x;
    int x = 0;
    if (initial) {
	assert(blockIdx.x==0);
    }
    else {
	x = fftwidth*(blockIdx.x+1); // second invocation: run nchan*nfft/cwidth-1 blocks.
    }

    if (t>=nchan) return;
    int block_width = blockDim.x;

    // blockIdx.y: index of first vector (2*antennaindex+polindex)

    int s = nfft*fftwidth;

    int i = blockIdx.y;
    int j0 = 2*(i/2+1);

    int ai = i/2;
    int b = 4*(ai*nant-ai*(ai+1)/2) + 2*(i-2*ai);

    const float2* iv = ants+i*s+t;
    const float2* jv = ants+j0*s+t;

    float2 u = iv[0];
    int hoff = threadIdx.x;
    for (int j = j0; j<2*nant; ++j) {
	float2 v = jv[0];
	float2 z;
	z.x = u.x*v.x + u.y*v.y;
	z.y = u.y*v.x - u.x*v.y;
	h[hoff] = z;

	jv += s;
	hoff += block_width;
    }

    for (int k = fftwidth; k<s; k += fftwidth) {
	u = iv[k];
	jv = ants+j0*s+t;

	int hoff = threadIdx.x;
	for (int j = j0; j<2*nant; ++j) {
	    float2 v = jv[k];
	    float2 z;
	    z.x = u.x*v.x + u.y*v.y;
	    z.y = u.y*v.x - u.x*v.y;
	    h[hoff].x += z.x;
	    h[hoff].y += z.y;

	    jv += s;
	    hoff += block_width;
	}
    }

    float oonfft = 1.f/nfft;

    hoff = threadIdx.x;
    for (int j = j0; j<2*nant; ++j) {
	float2 a = h[hoff];
	a.x *= oonfft;
	a.y *= oonfft;

	int dj = j-j0;
	int aj = 2*(dj/2);
	accum[(b+aj*2+(dj-aj))*nchan+t] = a;
	hoff += block_width;
    }
template <int pj>
__global__ void CCAH3(cuComplex *accum, const cuComplex *ants, int nant, int nfft, int nchan, int fftwidth) {
}
#endif

std::vector<float2> run_CCAH3(const std::vector<float2>& data, int nant, int nfft, int nchan, int fftwidth) {
    constexpr int npol = 2;
    int nantxp = nant*npol;

    int block_width = 128;
    dim3 ccblock(nblocks(nchan, block_width), nantxp-1);

    size_t result_sz = nant*(nant-1)/2*npol*npol*nchan;

    gpu_array<float2> gpu_data(data);
    gpu_array<float2> gpu_result(result_sz);

    int shared_alloc =sizeof(float2)*(nantxp-2)*block_width;
    CCAH3<<<ccblock, block_width, shared_alloc>>>(gpu_result.data(), gpu_data.data(), nant, nfft, nchan, fftwidth);
    return gpu_result;
}

double time_CCAH3(int repeat_count, const float2* gpu_data, int nant, int nfft, int nchan, int fftwidth) {
    constexpr int npol = 2;
    int nantxp = nant*npol;

    int block_width = 128;
    dim3 ccblock(nblocks(nchan, block_width), nantxp-1);

    size_t result_sz = nant*(nant-1)/2*npol*npol*nchan;
    gpu_array<float2> gpu_result(result_sz);

    int shared_alloc =sizeof(float2)*(nantxp-2)*block_width;
    return run_kernel(repeat_count,
	[&]() {
	    CCAH3<<<ccblock, block_width, shared_alloc>>>(gpu_result.data(), gpu_data, nant, nfft, nchan, fftwidth);
	});
}

std::vector<float2> run_CrossCorr(const std::vector<float2>& data, int nant, int nfft, int nchan, int fftwidth) {
    int targetThreads = 50e4;
    int parallelAccum = (int)ceil(targetThreads/nchan+1);
    while (parallelAccum && nfft % parallelAccum) parallelAccum--;

    int block_width = 512;
    int blockx = nblocks(nchan, block_width);

    dim3 corrBlocks(blockx, parallelAccum);
    dim3 accumBlocks(blockx, 4, nant*(nant-1)/2);

    size_t result_sz = nant*(nant-1)*2*nchan;

    gpu_array<float2> gpu_data(data);
    gpu_array<float2> gpu_baselinedata(result_sz*parallelAccum);

    int nchunk = nfft/parallelAccum;
    CrossCorr<<<corrBlocks, block_width>>>(gpu_data.data(), gpu_baselinedata.data(), nant, nchunk);
    finaliseAccum<<<accumBlocks, block_width>>>(gpu_baselinedata.data(), parallelAccum, nchunk);

    std::vector<float2> baselinedata(gpu_baselinedata);
    std::vector<float2> result(result_sz);

    int nvec = nant*(nant-1)*2;
    int rstride = nchan, bstride = nchan*parallelAccum;
    for (int i = 0; i<nvec; ++i) {
        std::copy(baselinedata.data()+i*bstride, baselinedata.data()+i*bstride+nchan, result.data()+i*rstride);
    }

    return result;
}

double time_CrossCorr(int repeat_count, const float2* gpu_data, int nant, int nfft, int nchan, int fftwidth) {
    int targetThreads = 50e4;
    int parallelAccum = (int)ceil(targetThreads/nchan+1);
    while (parallelAccum && nfft % parallelAccum) parallelAccum--;

    int block_width = 512;
    int blockx = nblocks(nchan, block_width);

    dim3 corrBlocks(blockx, parallelAccum);
    dim3 accumBlocks(blockx, 4, nant*(nant-1)/2);

    size_t result_sz = nant*(nant-1)*2*nchan;
    gpu_array<float2> gpu_baselinedata(result_sz*parallelAccum);

    return run_kernel(repeat_count,
	[&]() {
            CrossCorr<<<corrBlocks, block_width>>>((float2*)gpu_data, gpu_baselinedata.data(), nant, nfft/parallelAccum);
            finaliseAccum<<<accumBlocks, block_width>>>(gpu_baselinedata.data(), parallelAccum, nfft/parallelAccum);
        });
}

// No unit test for FringeRotate routines yet.

double time_FringeRotate(int repeat_count, const float2* gpu_data, const float* gpu_rotvec, int nant, int nfft, int fftwidth) {
    int block_width = 512;
    dim3 fringeBlocks(nblocks(fftwidth, block_width), nfft);

    return run_kernel(repeat_count,
	[&]() {
            FringeRotate<<<fringeBlocks, block_width>>>((float2*)gpu_data, (float*)gpu_rotvec);
        });
}

double time_FringeRotate2(int repeat_count, const float2* gpu_data, const float* gpu_rotvec, int nant, int nfft, int fftwidth) {
    int block_width = 512;
    dim3 fringeBlocks(nblocks(fftwidth, block_width), nfft);

    return run_kernel(repeat_count,
	[&]() {
            FringeRotate2<<<fringeBlocks, block_width>>>((float2*)gpu_data, (float*)gpu_rotvec);
        });

}
