#include <algorithm>

template <typename T>
void ConcatenateArrays(T * array1, int size1, T * array2, int size2,
		T * & result, int & resultSize)
{
	resultSize = size1 + size2;
	result = new T[resultSize];
	std::copy(array1, array1 + size1, result);
	std::copy(array2, array2 + size2, result + size1);
}

int GetLowestPowerOf2GreaterThanN(int N)
{
	double logBase2 = log(N)/log(2);
	int powerOf2 = int(logBase2 + 1.0);
	return pow(2,powerOf2);
}