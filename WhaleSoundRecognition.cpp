/*
 * WhaleSoundRecognition.cpp
 *
 *  Created on: 24 Mar 2013
 *      Author: michael
 */

#include "WhaleSoundRecognition.h"
#include "FeatureVector.h"

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

FeatureVector WhaleSoundRecognition::GetFeatures(double * samples, int nSamples)
{
	// For the audio file consisting of the samples passed in get the features
	// to describe it.
	//fftw_complex * out;
	//GetPeriodogramEstimate(samples, nSamples, out);

	// Do mfcc thing and extract whatever other features we want.
	//xtract_spectrum()
	//xtract_mfcc()

	//fftw_free(out);
	float * data = new float[100];
	FeatureVector f(data, 100);
	delete [] data;
	return f;
}
