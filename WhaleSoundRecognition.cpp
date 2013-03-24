/*
 * WhaleSoundRecognition.cpp
 *
 *  Created on: 24 Mar 2013
 *      Author: michael
 */

#include "WhaleSoundRecognition.h"
#include "FeatureVector.h"

#include <iostream>
#include <cassert>
#define LIBAIFF_NOCOMPAT 1
extern "C"
{
#include <libaiff/libaiff.h>
}

#include "boost/filesystem.hpp"
#include "boost/regex.hpp"

using namespace boost::filesystem;
using namespace std;

WhaleSoundRecognition::WhaleSoundRecognition() {
	// TODO Auto-generated constructor stub

}

WhaleSoundRecognition::~WhaleSoundRecognition() {
	// TODO Auto-generated destructor stub
}

void WhaleSoundRecognition::Init(const string & pathToTrainData,
		const string & pathToTestData)
{
	pathToTrainingData_ = pathToTrainData;
	pathToTestData_ = pathToTestData;
}

void WhaleSoundRecognition::Train()
{
	// Get all filenames ending .aiff
	path current_dir(pathToTrainingData_);
	boost::regex pattern("train[0-9]+.aiff");
	for (recursive_directory_iterator iter(current_dir), end;
			iter != end;
			++iter)
	{
		string name = iter->path().leaf().string();
		if (regex_match(name, pattern))
			AddTrainingData(iter->path().string());
	}
}

void WhaleSoundRecognition::AddTrainingData(const std::string & filename)
{
	double * samples;
	int nSamples;
	GetRawSamples(filename, samples, nSamples);

	GetFeatures(samples, nSamples);
	delete [] samples;
}

// Caller should use delete [] samples
void WhaleSoundRecognition::GetRawSamples(const std::string & filename,
		double * & samples, int & nSamples)
{
	AIFF_Ref ref = AIFF_OpenFile(filename.c_str(), F_RDONLY);

	nSamples = ref->nSamples;
	// Our AIFF files have audio format LPCM and 16 bits per sample.
	int16_t * samplesI = new int16_t[nSamples];
	int numBytesRead = AIFF_ReadSamples(ref,samples,nSamples*ref->bitsPerSample/8) ;
	assert(numBytesRead == nSamples*ref->bitsPerSample/8);
	// Convert to doubles to use with the fftw
	samples = new double[nSamples];
	for (size_t i=0; i<nSamples; ++i)
	{
		samples[i] = (double)samplesI[i];
	}
	delete [] samplesI;
}

// Caller should use fftw_free(out)
void WhaleSoundRecognition::GetPeriodogramEstimate(double * samples, int nSamples,
		fftw_complex * & out)
{
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nSamples);
	fftw_plan p = fftw_plan_dft_r2c_1d(nSamples, samples, out,
			FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);
}

FeatureVector WhaleSoundRecognition::GetFeatures(double * samples, int nSamples)
{
	// For the audio file consisting of the samples passed in get the features
	// to describe it.
	fftw_complex * out;
	GetPeriodogramEstimate(samples, nSamples, out);

	// Do mfcc thing and extract whatever other features we want.

	fftw_free(out);
	return FeatureVector();
}
