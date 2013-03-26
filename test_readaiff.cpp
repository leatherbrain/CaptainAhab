/***************************************************************************
 * FILE:			test_readaiff.cpp
 * DESCRIPTION:		
 * 
 * CREATED ON:		21 Mar 2013
 * AUTHOR:			S. Sundaram
 ***************************************************************************/

#include <iostream>
#include <libaiff/libaiff.h>

#define LIBAIFF_NOCOMPAT 1

int main()
{
	std::string filename = "/home/realvis/dungeon/whales/data/wav_train/train1.aiff";
	AIFF_Ref soundFile = AIFF_OpenFile(filename.c_str(), F_RDONLY);
	if (!soundFile)
	{
		std::cout << "Kill yourselves. Now." << std::endl;
		return 1;
	}

	// Attempt to read file
}
