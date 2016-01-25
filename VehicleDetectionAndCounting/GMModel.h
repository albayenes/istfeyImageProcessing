#include<opencv2/core/core.hpp>
using namespace cv;

class GMModel
{
private:
	int K;
	int height;
	int width;
	float alpha;
	float initStd;
	void updateMinWeightStatistics(Mat frame, int h, int w, bool match);
public:
	Mat *weightMatrices;
	Mat *stdMatrices;
	Mat *meanMatrices;
	Mat *diffFrame;
	Mat foreGround;
	Mat sumWeights;
	GMModel();
	GMModel(const int K, int height, int width);
	void findForeGround(Mat frame);
};