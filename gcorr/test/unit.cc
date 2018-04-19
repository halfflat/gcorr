#include <cstddef>
#include <vector>
#include <iostream>
#include <random>

#include <cuComplex.h>

#include "gtest.h"
#include "wrappers.h"
#include "mkdata.h"

using std::size_t;

bool operator==(float2 a, float2 b) {
    return a.x==b.x && a.y==b.y;
}

bool operator!=(float2 a, float2 b) {
    return !(a==b);
}

std::ostream& operator<<(std::ostream& o, float2 v) {
    return o << '(' << v.x << ',' << v.y << ')';
}

// Use complex small magnitude ints to avoid rouding artefacts in testing.
std::vector<float2> random_cint_data(size_t n, int seed = 12345) {
    std::minstd_rand R(seed);
    std::uniform_int_distribution<int> U(-5, 5);

    std::vector<float2> data(n);
    std::generate(data.begin(), data.end(), [&]() {
	float2 f;
	f.x = U(R);
	f.y = U(R);
       	return f;
    });
    return data;
}

template <int npol>
std::vector<float2> expected_crosscorraccum(const std::vector<float2>& data, int nant, int nfft, int nchan, int fftwidth) {
    size_t result_sz = nant*(nant-1)/2*npol*npol*nchan;
    std::vector<float2> result(result_sz);

    size_t stride = nfft*fftwidth;
    size_t b = 0;

    for (int i = 0; i<nant-1; ++i) {
	for (int j = i+1; j<nant; ++j) {
	    for (int pi = 0; pi<npol; ++pi) {
		for (int pj = 0; pj<npol; ++pj) {
		    for (int k = 0; k<nchan; ++k) {
			float2 a = {0.f, 0.f};
			for (int f = 0; f<nfft; ++f) {
			    float2 u = data[(pi+i*npol)*stride+k+f*fftwidth];
			    float2 v = data[(pj+j*npol)*stride+k+f*fftwidth];
			    a.x += u.x*v.x + u.y*v.y;
			    a.y += u.y*v.x - u.x*v.y;
			}
			a.x /= nfft;
			a.y /= nfft;
			result[b++] = a;
		    }
		}
	    }
	}
    }

    return result;
}

TEST(gxkernel, CrossCorrAccumHoriz) {
    constexpr int pol = 2;

    {
	int nant = 3;
	int fftwidth = 512;
	int nchan = 256;
	int nfft = 4;

        size_t datasz = pol*nant*fftwidth*nfft;
	auto data = random_cint_data(datasz, -5, 5);

	auto expected = expected_crosscorraccum<pol>(data, nant, nfft, nchan, fftwidth);
	auto result = run_CrossCorrAccumHoriz(data, nant, nfft, nchan, fftwidth);
	EXPECT_EQ(expected, result);
	for (int i=0; i<result.size(); ++i) {
	    ASSERT_EQ(expected[i], result[i]) << "unequal at i=" << i;
	}
    }
    {
	int nant = 5;
	int fftwidth = 1024;
	int nchan = 1024;
	int nfft = 100;

        size_t datasz = pol*nant*fftwidth*nfft;
	auto data = random_cint_data(datasz, -5, 5);

	auto expected = expected_crosscorraccum<pol>(data, nant, nfft, nchan, fftwidth);
	auto result = run_CrossCorrAccumHoriz(data, nant, nfft, nchan, fftwidth);
	EXPECT_EQ(expected, result);
    }
}

TEST(gxkernel, CCCAH2) {
    constexpr int pol = 2;

    {
	int nant = 3;
	int fftwidth = 512;
	int nchan = 256;
	int nfft = 4;

        size_t datasz = pol*nant*fftwidth*nfft;
	auto data = random_cint_data(datasz, -5, 5);

	auto expected = expected_crosscorraccum<pol>(data, nant, nfft, nchan, fftwidth);
	auto result = run_CCAH2(data, nant, nfft, nchan, fftwidth);
	EXPECT_EQ(expected, result);
	for (int i=0; i<result.size(); ++i) {
	    ASSERT_EQ(expected[i], result[i]) << "unequal at i=" << i;
	}
    }
    {
	int nant = 5;
	int fftwidth = 1024;
	int nchan = 1024;
	int nfft = 100;

        size_t datasz = pol*nant*fftwidth*nfft;
	auto data = random_cint_data(datasz, -5, 5);

	auto expected = expected_crosscorraccum<pol>(data, nant, nfft, nchan, fftwidth);
	auto result = run_CCAH2(data, nant, nfft, nchan, fftwidth);
	EXPECT_EQ(expected, result);
    }
}


TEST(gxkernel, CrossCorr) {
    // require nchan*2=fftwidth and nchan a multiple of 512.
    constexpr int pol = 2;

    {
	int nant = 3;
	int fftwidth = 1024;
	int nchan = 512;
	int nfft = 3000;

        size_t datasz = pol*nant*fftwidth*nfft;
	auto data = random_cint_data(datasz, -5, 5);

	auto expected = expected_crosscorraccum<pol>(data, nant, nfft, nchan, fftwidth);
	auto result = run_CrossCorr(data, nant, nfft, nchan, fftwidth);
	EXPECT_EQ(expected, result);
    }
}

struct packed_set {
    std::vector<int8_t> packed;
    std::vector<int32_t> shifts;
    int packed_offset;
    int packed_stride;
};

packed_set generate_random_packed_data(int nant, int nfft, int fftwidth) {
    packed_set P;

    constexpr int npol = 2;
    constexpr int max_shift = 4;

    P.shifts = random_int_data(nant*nfft, -max_shift, max_shift);
    for (auto& s: P.shifts) s = 2*(s/2); // restrict to even

    int samp_per_antenna = (2*max_shift+nfft*fftwidth/2)*npol;
    P.packed_stride = samp_per_antenna/4+1;  // 2 bits per sample
    P.packed_offset = (max_shift*npol)/4;

    P.packed = random_int8_data(P.packed_stride*nant, -128, 127);
    return P;
}

std::vector<float2> expected_unpack2bit_2chan(const packed_set& P, int nant, int nfft, int fftwidth) {
    constexpr float HiMag = 3.3359;
    const float lut4level[4] = {-HiMag, -1.0, 1.0, HiMag};

    int s = nfft*fftwidth;
    std::vector<float2> unpacked(nant*2*s);
    for (int i = 0; i<nant; ++i) {
        int unpacked_base = i*2*s;

        for (int j = 0; j<nfft; ++j) {
            int shift = P.shifts[i*nfft+j];
            int packed_base = P.packed_offset + (i*P.packed_stride - shift)/2;
            unpacked_base += fftwidth;

            for (int k = 0; k<fftwidth/2; k+=2) {
                unsigned char b = P.packed[packed_base+k/2];
                float2 a00 = {lut4level[b&3], 0};
                b>>=2;
                float2 a10 = {lut4level[b&3], 0};
                b>>=2;
                float2 a01 = {lut4level[b&3], 0};
                b>>=2;
                float2 a11 = {lut4level[b&3], 0};

                unpacked[k] = a00;
                unpacked[k+1] = a01;
                unpacked[s+k] = a10;
                unpacked[s+k+1] = a11;
            }
        }
    }
    return unpacked;
}

TEST(gxkernel, unpack2bit_2chan_fast) {
    int nant = 6;
    int nfft = 3000;
    int fftwidth = 1024;

    packed_set P = generate_random_packed_data(nant, nfft, fftwidth);

    auto expected = expected_unpack2bit_2chan(P, nant, nfft, fftwidth);
    auto result = run_unpack2bit_2chan_fast(P.packed, P.packed_stride, P.packed_offset, P.shifts, nant, nfft, fftwidth);
    EXPECT_EQ(expected, result);
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

