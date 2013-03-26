/*
 * WhaleSoundRecognition.cpp
 *
 *  Created on: 24 Mar 2013
 *      Author: michael
 */

#include "WhaleSoundRecognition.h"
#include "FeatureVector.h"
#include "xtract/libxtract.h"

#include <iostream>
#include <fstream>
#include <cassert>
#define LIBAIFF_NOCOMPAT 1
extern "C"
{
#include <libaiff/libaiff.h>
}
#include "xtract/libxtract.h"

#include "boost/filesystem.hpp"
#include "boost/regex.hpp"

using namespace boost::filesystem;

using namespace std;

WhaleSoundRecognition::WhaleSoundRecognition() {
	// TODO Auto-generated constructor stub
	statModel = new CvSVM();
}

WhaleSoundRecognition::~WhaleSoundRecognition() {
	// TODO Auto-generated destructor stub
	delete statModel;
}

void WhaleSoundRecognition::Init(const string & pathToTrainData,
		const string & trainDataLabelsFilename,
		const string & pathToTestData)
{
	pathToTrainingData_ = pathToTrainData;
	pathToTestData_ = pathToTestData;
	trainDataLabelsFilename_ = trainDataLabelsFilename;
}

void WhaleSoundRecognition::Train()
{
	trainingData.clear();

	ifstream trainDataFile(trainDataLabelsFilename_.c_str(), ios_base::in);

	string currentLine;
	getline(trainDataFile, currentLine);
	while (trainDataFile.good() && !trainDataFile.eof())
	{
		getline(trainDataFile, currentLine);
		if (trainDataFile.good() && !trainDataFile.eof())
		{
			cout << currentLine << endl;
			int commaPos = currentLine.find_first_of(',');
			string filename = currentLine.substr(0,commaPos);
			bool containsWhaleSound = currentLine[commaPos+1] == '1';
			AddTrainingData(pathToTrainingData_ + filename, containsWhaleSound);
		}
	}

	if (trainingData.empty())
	{
		return;
	}

	cv::Mat allTrainingData(trainingData.size(), trainingData.front().data_.cols, CV_32FC1);
	cv::Mat allTrainingLabels(trainingData.size(), 1, CV_32SC1);
	for (size_t i=0; i<trainingData.size(); ++i)
	{
		trainingData[i].data_.copyTo( allTrainingData(cv::Range(i, i+1), cv::Range::all()));
		allTrainingLabels.at<int>(i,0) = trainingDataLabels[i] ? 1 : 0;
	}

	statModel->train(allTrainingData, allTrainingLabels);

}

void WhaleSoundRecognition::Test()
{
	// Get all filenames ending .aiff
	path current_dir(pathToTestData_);
	boost::regex pattern("test[0-9]+.aiff");
	for (recursive_directory_iterator iter(current_dir), end;
			iter != end;
			++iter)
	{
		string name = iter->path().leaf().string();
		if (regex_match(name, pattern))
		{
			double * samples;
			int nSamples;
			GetRawSamples(iter->path().string(), samples, nSamples);
			FeatureVector f = GetFeatures(samples, nSamples);
			float result = statModel->predict(f.data_);
			cout << "result = " << result << endl;
		}
	}
}

void WhaleSoundRecognition::AddTrainingData(const std::string & filename,
		bool containsWhaleSound)
{
	double * samples;
	int nSamples;
	GetRawSamples(filename, samples, nSamples);

	FeatureVector f = GetFeatures(samples, nSamples);
	trainingData.push_back(f);
	trainingDataLabels.push_back(containsWhaleSound);
	cout << filename << " " << containsWhaleSound << endl;
	delete [] samples;
}

// Caller should use delete [] samples
void WhaleSoundRecognition::GetRawSamples(const std::string & filename,
		double * & samples, int & nSamples)
{
	AIFF_Ref ref = AIFF_OpenFile(filename.c_str(), F_RDONLY);

	if (!ref)
	{
		cout << "Failed to open " << filename << endl;
		return;
	}

	nSamples = ref->nSamples;

	// Our AIFF files have audio format LPCM and 16 bits per sample.
	int16_t * samplesI = new int16_t[nSamples];
	int numBytesRead = AIFF_ReadSamples(ref,samplesI,nSamples*ref->bitsPerSample/8) ;
	assert(numBytesRead == nSamples*ref->bitsPerSample/8);
	// Convert to doubles to use with the fftw
	samples = new double[nSamples];
	for (int i=0; i<nSamples; ++i)
	{
		samples[i] = (double)samplesI[i];
	}
	delete [] samplesI;

	AIFF_CloseFile(ref);
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

FeatureVector WhaleSoundRecognition::GetFeatures(double *input, int nSamples)
{
    /* get the mean of the input */
	double mean;
    xtract[XTRACT_MEAN](input, nSamples, NULL, &mean);

    /* get the spectrum */
    double argd[4];
    const int SAMPLERATE = 44100;
    double spectrum[nSamples];
    argd[0] = SAMPLERATE / (double)nSamples;
    argd[1] = XTRACT_MAGNITUDE_SPECTRUM;
    argd[2] = 0.f; /* No DC component */
    argd[3] = 0.f; /* No Normalisation */

    xtract_init_fft(nSamples, XTRACT_SPECTRUM);
    xtract[XTRACT_SPECTRUM](input, nSamples, &argd[0], spectrum);

    /* compute the MFCCs */
    const int MFCC_FREQ_BANDS = 13;
    int n;
    xtract_mel_filter mel_filters;
    mel_filters.n_filters = MFCC_FREQ_BANDS;
    mel_filters.filters   = (double **)malloc(MFCC_FREQ_BANDS * sizeof(double *));
    for(n = 0; n < MFCC_FREQ_BANDS; ++n)
    {
        mel_filters.filters[n] = (double *)malloc(nSamples * sizeof(double));
    }

    const int MFCC_FREQ_MIN = 20;
    const int MFCC_FREQ_MAX = 20000;
    double mfccs[MFCC_FREQ_BANDS * sizeof(double)];
    xtract_init_mfcc(nSamples >> 1, SAMPLERATE >> 1, XTRACT_EQUAL_GAIN, MFCC_FREQ_MIN, MFCC_FREQ_MAX, mel_filters.n_filters, mel_filters.filters);
    xtract_mfcc(spectrum, nSamples >> 1, &mel_filters, mfccs);

	//fftw_free(out);
	float * data = new float[100];
	FeatureVector f(data, 100);
	delete [] data;
	return f;
}
