/*
 * WhaleSoundRecognition.h
 *
 *  Created on: 24 Mar 2013
 *      Author: michael
 */

#ifndef WHALESOUNDRECOGNITION_H_
#define WHALESOUNDRECOGNITION_H_

#include <string>
#include <vector>
#include <fftw3.h>
#include <opencv/ml.h>

class FeatureVector;

class WhaleSoundRecognition {
public:
	WhaleSoundRecognition();
	void Init(const std::string & pathToTrainData,
			const std::string & trainDataLabelsFilename,
			const std::string & pathToTestData);
	virtual ~WhaleSoundRecognition();
	void Train();
	void MockTest();
	void Test();

private:
	void AddTrainingData(const std::string & filename,
			bool containsWhaleSound);
	void AddTestingData(const std::string & filename,
			bool containsWhaleSound);
	void GetRawSamples(const std::string & filename,
			double * & samples, int & nSamples);
	void GetPeriodogramEstimate(double * samples, int nSamples,
			fftw_complex * & out);
	float * GetWholeClipFeatures(const std::string& filename);
	void GetSpectrum(const int SAMPLERATE, int nSamples, double* input,
			double * spectrum);
	const int ExtractMFCCs(int nSamples, const int SAMPLERATE,
			double *spectrum, double*& mfccs);
	double CalculateLogEnergy(double * input, int nSamples);
	std::vector<float *> GetPerFrameFeatures(const std::string& filename);
	float * GetHistogramOfFrameFeatures(const std::vector<float *> & f);

	std::string pathToTrainingData_, pathToTestData_, trainDataLabelsFilename_;

	std::vector<FeatureVector> trainingData;
	std::vector<bool> trainingDataLabels;
	std::vector<std::vector<float *> > soundClipsPerFrameVector;
	std::vector<float *> soundClipsWholeClipVector;
	cv::Mat centres;

	std::vector<std::string> mockTestFilenames;
	std::vector<bool> mockTestGroundTruth;

	// Wanted to make this a CvStatModel but this base class does not declare
	// the train method interface.
	//CvSVM * statModel;
	CvRTrees * statModel;
	CvRTParams * params;
};

#endif /* WHALESOUNDRECOGNITION_H_ */
