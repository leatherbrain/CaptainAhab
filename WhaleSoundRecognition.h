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
	void Test();

private:
	void AddTrainingData(const std::string & filename,
			bool containsWhaleSound);
	void GetRawSamples(const std::string & filename,
			double * & samples, int & nSamples);
	void GetPeriodogramEstimate(double * samples, int nSamples,
			fftw_complex * & out);
	FeatureVector GetFeatures(double * samples, int nSamples);
	std::string pathToTrainingData_, pathToTestData_, trainDataLabelsFilename_;

	std::vector<FeatureVector> trainingData;
	std::vector<bool> trainingDataLabels;

	// Wanted to make this a CvStatModel but this base class does not declare
	// the train method interface.
	CvSVM * statModel;
};

#endif /* WHALESOUNDRECOGNITION_H_ */
