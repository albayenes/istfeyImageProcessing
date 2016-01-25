#include <iostream>
#include "GMModel.h"

// Initialize GMM
// K is the number of Gauss
GMModel::GMModel(const int K, int height, int width)
{
	this->K = K;
	this->height = height;
	this->width = width;
	this->alpha = 0.1f;
	// Initialize weight matrix
	//Mat **weightMatrices = new Mat*[K];
	weightMatrices = new Mat[K];
	for (int i = 0; i < K; i++){
		weightMatrices[i] = Mat(height, width, CV_32FC1, Scalar(1.0 / K));
	}

	// initialize deviation matrix
	this->initStd = 6 / 256.0;
	stdMatrices = new Mat[K];
	for (int i = 0; i < K; i++){
		stdMatrices[i] = Mat(height, width, CV_32FC1, Scalar(initStd));
	}

	// initialize mean matrix
	meanMatrices = new Mat[K];
	for (int i = 0; i < K; i++){
		meanMatrices[i] = Mat(height, width, CV_32FC1);
		randu(meanMatrices[i], Scalar::all(0.0), Scalar::all(1.0));
	}

	// difference matrix
	diffFrame = new Mat[K];

	// sum of weights
	sumWeights = Mat(height, width, CV_32FC1, Scalar(0.0));
}

void GMModel::findForeGround(Mat frame)
{
	// foregournd matrix
	foreGround = Mat(height, width, CV_8UC1, Scalar(0));

	
	for (int h = 0; h < height; h++){
		for (int w = 0; w < width; w++){
			bool match = false;
			for (int i = 0; i < K; i++)
			{
				
				/*std::cout << abs(this->meanMatrices[i].at<float>(h, w) - frame.at<float>(h, w)) << std::endl;
				std::cout << this->meanMatrices[i].at<float>(h, w) << std::endl;
				std::cout << (3 * this->stdMatrices[i].at<float>(h, w)) << std::endl;
				std::cout << "********************************" << std::endl;*/

				if (abs(this->meanMatrices[i].at<float>(h, w) - frame.at<float>(h, w)) < (3.5f * this->stdMatrices[i].at<float>(h, w)))
				{
					// weight updated if there is match
					this->weightMatrices[i].at<float>(h, w) = this->weightMatrices[i].at<float>(h, w) * (1.0f - alpha) + alpha;
					float expPart = exp(-1.0f * pow((frame.at<float>(h, w) - this->meanMatrices[i].at<float>(h, w)), 2.0f) / (2.0f * pow(this->stdMatrices[i].at<float>(h, w), 2.0f)));
					float firstPart = sqrt(1.0f / (2.0f * 3.14f))*pow(this->stdMatrices[i].at<float>(h, w), -1.0f);
					float multiFirstExp = expPart * firstPart;
					float tempAlpha = alpha / multiFirstExp;

					// update mean and standart deviation if there is match
					this->meanMatrices[i].at<float>(h, w) = (1.0f - tempAlpha)*this->meanMatrices[i].at<float>(h, w) + tempAlpha*frame.at<float>(h, w);
					this->stdMatrices[i].at<float>(h, w) = sqrt((1.0f - tempAlpha)*pow(this->stdMatrices[i].at<float>(h, w), 2.0f) + tempAlpha * pow((frame.at<float>(h, w) - this->meanMatrices[i].at<float>(h, w)), 2.0f));

					match = true;
					break;
				}
				else
				{
					this->weightMatrices[i].at<float>(h, w) = this->weightMatrices[i].at<float>(h, w) * (1.0f - alpha);
				}
			}

			if (match == false)
			{
				//std::cout << "h: " << h << ", w: " << w << std::endl;
				this->foreGround.at<unsigned char>(h, w) = 255;
				updateMinWeightStatistics(frame, h, w, match);
			}

		}
	}
}

void GMModel::updateMinWeightStatistics(Mat frame, int h, int w, bool match)
{
	
		int minWeightID = 0;
		for (int i = 1; i < K; i++)
		{
			if (this->weightMatrices[i].at<float>(h, w) < this->weightMatrices[minWeightID].at<float>(h, w))
			{
				minWeightID = i;
			}
		}

		this->meanMatrices[minWeightID].at<float>(h, w) = frame.at<float>(h, w);
		this->stdMatrices[minWeightID].at<float>(h, w) = this->initStd;
	
}