//============================================================================
// Name        : CaptainAhab.cpp
// Author      : Michael King and Sudeep Sundaram
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cassert>
#define LIBAIFF_NOCOMPAT 1
extern "C"
{
#include <libaiff/libaiff.h>
}
#include <fftw3.h>
using namespace std;

int main() {
	AIFF_Ref ref = AIFF_OpenFile("/home/michael/Documents/Kaggle/Whale sounds/data/train/train1.aiff", F_RDONLY);

	// Our AIFF files have audio format LPCM and 16 bits per sample.
	int16_t * samples = new int16_t[ref->nSamples];
	int numBytesRead = AIFF_ReadSamples(ref,samples,ref->nSamples*ref->bitsPerSample/8) ;
	assert(numBytesRead == ref->nSamples*ref->bitsPerSample/8);
	// Convert to doubles to use with the fftw
	double * samplesD = new double[ref->nSamples];
	for (size_t i=0; i<ref->nSamples; ++i)
	{
		samplesD[i] = (double)samples[i];
	}
	fftw_complex * out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ref->nSamples);
	fftw_plan p = fftw_plan_dft_r2c_1d(ref->nSamples, samplesD, out,
			FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
	fftw_free(out);
	return 0;
}
