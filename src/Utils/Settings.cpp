/********************
      Settings
********************/

#include "Settings.hpp"

Settings::Settings() {
  width         = 1280;
  height        = 800;
  max_iters     = 256;
  autoZoom      = false;
  offsetX       = -0.7;
  offsetY       = 0.0;
  zoom          = 0.004;
  finalZoom     = 0.004;
  zoomFactor    = 0.01;
  zoomStepTime  = 0.01;
  nbSimulations = 1;
  nbCudaThreads = 1;

  isCentralDotEnabled  = false;
  closeAfterSimulation = false;
  logTofile            = false;
  testID               = 0;
  firstSimulation      = false;
  lastSimulation       = false;
  isCuteModeEnabled    = false;

  convergenceType = "DP_OMP_AVXPLUS";
  colorMapMode    = "SMOOTH_SHIFTED";
}

void Settings::SetSize(int w, int h) {
  width  = w;
  height = h;
}

void Settings::Width(int w) {
    width = w;
}

void Settings::Height(int h) {
    height = h;
}

int Settings::Width() {
    return width;
}

int Settings::Height() {
    return height;
}

std::string Settings::ColorMapMode() {
    return colorMapMode;
}

void Settings::ColorMapMode(std::string type) {
    colorMapMode = type;
}

void Settings::SetOffsetX(double ox) {
  offsetX = ox;
}

void Settings::SetOffsetY(double oy) {
  offsetY = oy;
}

void Settings::Zoom(double z) {
    Settings::zoom = z;
}

double Settings::Zoom() {
    return zoom;
}

void Settings::SetFinalZoom(double z) {
  Settings::finalZoom = z;
}

void Settings::SetZoomFactor(double zf) {
  Settings::zoomFactor = zf;
}

void Settings::SetZoomStepTime(double zts) {
  Settings::zoomStepTime = zts;
}

void Settings::SetAutoZoom(bool az) {
  Settings::autoZoom = az;
}

void Settings::SetNbSimulations(unsigned int nbSimu) {
  Settings::nbSimulations = nbSimu;
}

void Settings::SetNbCudaThreads(unsigned int nbCudaTh) {
  Settings::nbCudaThreads = nbCudaTh;
}

void Settings::SetCentralDot(bool cd) {
  Settings::isCentralDotEnabled = cd;
}

void Settings::Iterations(int mi) {
    Settings::max_iters = mi;
}

int Settings::Iterations() {
    return max_iters;
}

std::string Settings::ConvergenceType() {
    return convergenceType;
}

void Settings::ConvergenceType(std::string type) {
    convergenceType = type;
}

void Settings::SetCloseAfterSimulation(bool cas) {
  Settings::closeAfterSimulation = cas;
}

void Settings::SetLogToFile(bool ltf) {
  Settings::logTofile = ltf;
}

void Settings::SetTestID(unsigned int tid) {
  Settings::testID = tid;
}

void Settings::SetFirstSimulation(bool fs) {
  Settings::firstSimulation = fs;
}

void Settings::SetLastSimulation(bool ls) {
  Settings::lastSimulation = ls;
}

void Settings::SetCuteMode(bool cm) {
  Settings::isCuteModeEnabled = cm;
}
/*
unsigned int Settings::GetConvOffset() {
  if (Settings::convergenceType == ConvergenceType::DP) {
    return 0;
  } else if (Settings::convergenceType == ConvergenceType::DP_OMP) {
    return 1;
  } else if (Settings::convergenceType == ConvergenceType::DP_OMP_AVX) {
    return 2;
  } else if (Settings::convergenceType == ConvergenceType::DP_OMP_AVXPLUS) {
    return 3;
  } else if (Settings::convergenceType == ConvergenceType::DP_OMP_AVXPLUSPLUS) {
    return 4;
  } else if (Settings::convergenceType == ConvergenceType::SP) {
    return 5;
  } else if (Settings::convergenceType == ConvergenceType::SP_OMP) {
    return 6;
  } else if (Settings::convergenceType == ConvergenceType::SP_OMP_AVX) {
    return 7;
  } else if (Settings::convergenceType == ConvergenceType::SP_OMP_AVXPLUS) {
    return 8;
  //} else if (Settings::convergenceType == ConvergenceType::SP_OMP_AVXPLUSPLUS) {
  //  return 9;
  } else if (Settings::convergenceType == ConvergenceType::FP) {
    return 10;
  } else if (Settings::convergenceType == ConvergenceType::FP_OMP) {
    return 11;
  } else if (Settings::convergenceType == ConvergenceType::FP_OMP_SSE2) {
    return 12;
  } else if (Settings::convergenceType == ConvergenceType::CUDA_D) {
    return 14;
  } else if (Settings::convergenceType == ConvergenceType::CUDA_D2) {
    return 15;
  } else if (Settings::convergenceType == ConvergenceType::CUDA_D2_WP) {
    return 16;
  } else if (Settings::convergenceType == ConvergenceType::CUDA_S) {
    return 17;
  } else {
    return -1;
  }
}
  */
