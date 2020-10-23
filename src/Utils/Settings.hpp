/********************
      Settings
********************/

#ifndef SETTINGS_H_
#define SETTINGS_H_

#include <cstdio>
#include <cstdlib>
#include <string>

enum ConvergenceType{
	SP, SP_OMP, SP_OMP_AVX, SP_OMP_AVXPLUS, SP_OMP_AVXPLUSPLUS,
	DP, DP_OMP, DP_OMP_AVX, DP_OMP_AVXPLUS, DP_OMP_AVXPLUSPLUS,
	FP, FP_OMP, FP_OMP_SSE2, FP_OMP_AVX2,
	CUDA_D, CUDA_D2, CUDA_D2_WP, CUDA_S
};

enum ColorMapMode{
	GREYSCALE,
	HSV, HSV_LOOP, HSV_SHIFTED,
	SMOOTH, SMOOTH_LOOP, SMOOTH_SHIFTED,
	LERP_BLUE_RED, LERP_RED_BLUE, LERP_BLUE_PURPLE, LERP_PURPLE_BLUE,
	LSD
};

class Settings {
private:
	int width;
	int height;
    int max_iters;

    std::string convergenceType;
    std::string    colorMapMode;

public:

	bool autoZoom;

	long double offsetX;
	long double offsetY;
	long double zoom;
	long double finalZoom;
	long double zoomFactor;
	long double zoomStepTime;
	unsigned int nbSimulations;
	unsigned int nbCudaThreads;

	bool isCentralDotEnabled;
	bool closeAfterSimulation;
	bool logTofile;
	unsigned int testID;

	bool firstSimulation;
	bool lastSimulation;

	bool isCuteModeEnabled;


	Settings();

	void SetSize  (int w, int h);
    void Width (int w);
    void Height(int h);

    int  Width ();
    int  Height();

    void   Zoom(double z);
    double Zoom();

    void SetOffsetX(double ox);
	void SetOffsetY(double oy);
	void SetFinalZoom(double z);
	void SetZoomFactor(double z);
	void SetZoomStepTime(double z);
	void SetNbSimulations(unsigned int nbSimu);
	void SetNbCudaThreads(unsigned int nbCudaTh);
    void Iterations(int mi);
    int  Iterations();
	void SetAutoZoom(bool az);
	void SetCentralDot(bool cd);
	void SetCloseAfterSimulation(bool cas);
	void SetLogToFile      (bool ltf);
	void SetTestID(unsigned int tid);
	void SetFirstSimulation(bool fs);
	void SetLastSimulation (bool ls);
	void SetCuteMode       (bool cm);

    void ColorMapMode(std::string type);
    std::string ColorMapMode();

    void ConvergenceType(std::string type);
    std::string ConvergenceType();

	unsigned int GetConvOffset();

};

#endif /*SETTINGS_H_*/
