/*
 * WhaleSoundRecognition.h
 *
 *  Created on: 24 Mar 2013
 *      Author: michael
 */

#ifndef WHALESOUNDRECOGNITION_H_
#define WHALESOUNDRECOGNITION_H_

#include <string>
#include <fftw3.h>

class FeatureVector;

class WhaleSoundRecognition {
public:
	WhaleSoundRecognition();
	void Init(const std::string & pathToTrainData, const std::string & pathToTestData);
	virtual ~WhaleSoundRecognition();
	void Train();

private:
	void AddTrainingData(const std::string & filename);
	void GetRawSamples(const std::string & filename,
			double * & samples, int & nSamples);
	void GetPeriodogramEstimate(double * samples, int nSamples,
			fftw_complex * & out);
	FeatureVector GetFeatures(double * samples, int nSamples);
	std::string pathToTrainingData_, pathToTestData_;
};

#endif /* WHALESOUNDRECOGNITION_H_ */
