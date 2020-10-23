/**********************\
      File Handler
***********************

Author :
--------
Jonathan SAUSSEREAU

Description :
-------------
Text file parser

Changelog :
-----------
  - 2019/04/17 : Initial release
  - 2019/04/27 : Improved to handle complex parsing
  - 2019/10/21 : Adapted to fit fractal plotting
  - 2019/11/28 : Adapted to fit fractal plotting with sfml
*/

#include "FileHandler.hpp"

ConfigReader::ConfigReader() {
}

ConfigReader::ConfigReader(string filename) {
  Open(filename);
  debug = false;
}

ConfigReader::~ConfigReader() {
}

int ConfigReader::Open(string filename) {
  ConfigReader::filename = filename;
  if (filename.length() <= 4 || filename.substr(filename.length()-4) != ".ini") {
    filename+= ".ini";
  }
  fr.open(filename.c_str());
  if (!fr) {
    string test_filename = "../" + filename;
    fr.open(test_filename.c_str());
    if (!fr) {
      test_filename = "../" + test_filename;
      fr.open(test_filename.c_str());
      if (!fr) {
        cout << "\033[33m" << endl; //unix only
        cerr << "warning: cannot find \"" << filename << "\", file ignored" << endl;
        cout << "\033[0m"  << endl;
        return -1;
      }
    }
  }
  return 0;
}


void ConfigReader::ParseParams(Settings& params) {
  if (fr) {
    string line;
    int lineNb = 1;
    while (std::getline(fr, line)) {
      removeAllCharFromString(' ', &line);
      removeAllCharFromString(';', &line);
      removeAllCharFromString('\t', &line);
      removeAllCharFromString('\n', &line);
      removeAllCharFromString('\r', &line);
      stringToUppercase(&line);
      if (line[0] != '#' && line[0] != '/' && !line.empty()) { //skip empty lines and comment lines
        vector<string> content = splitString(line, "=");
        string identifier = content.at(0);
        if (content.size() > 1) {
          string arg = content.at(1);
          if (identifier == "WIDTH") {
              params.Width(stringToInt(arg));
          } else if (identifier == "HEIGHT") {
              params.Height(stringToInt(arg));

          } else if (identifier == "OFFSETX") {
              params.SetOffsetX(stringToDouble(arg));
          } else if (identifier == "OFFSETY") {
              params.SetOffsetY(stringToDouble(arg));
          } else if (identifier == "MAXITER") {
              params.Iterations(stringToInt(arg));
          } else if (identifier == "ZOOMFACTOR") {
              params.SetZoomFactor(stringToDouble(arg));
          } else if (identifier == "ZOOMSTEPTIME") {
              params.SetZoomStepTime(stringToDouble(arg));
          } else if (identifier == "STARTZOOM") {
              params.Zoom(stringToDouble(arg));
          } else if (identifier == "FINALZOOM") {
              params.SetFinalZoom(stringToDouble(arg));
          } else if (identifier == "NBSIMU") {
              params.SetNbSimulations(stringToInt(arg));
          } else if (identifier == "NBCUDATHREADS") {
              params.SetNbCudaThreads(stringToInt(arg));
          } else if (identifier == "AUTOZOOM") {
            if (arg == "TRUE" || arg == "ON") {
                params.SetAutoZoom(true);
            } else if (arg == "FALSE" || arg == "OFF") {
                params.SetAutoZoom(false);
            } else {
              cout << "\033[33m" << endl; //unix only
              cerr << "warning: unknown parameter \"" << arg << "\" at line " << lineNb << endl;
              cout << "\033[0m"  << endl;
            }
          } else if (identifier == "CONVERGENCE") {
            bool sucessfullyParsed = ParseConvergenceType(arg, params);
            if (!sucessfullyParsed) {
              cout << "\033[33m" << endl; //unix only
              cerr << "warning: unknown convergence type \"" << arg << "\" at line " << lineNb << endl;
              cout << "\033[0m"  << endl;
            }
          } else if (identifier == "COLORMAP") {
            bool sucessfullyParsed = ParseColorMap(arg, params);
            if (!sucessfullyParsed) {
              cout << "\033[33m" << endl; //unix only
              cerr << "warning: unknown colormap \"" << arg << "\" at line " << lineNb << endl;
              cout << "\033[0m"  << endl;
            }
          } else  {
            cout << "\033[33m" << endl; //unix only
            cerr << "warning: unknown identifier \"" << identifier << "\" at line " << lineNb << endl;
            cout << "\033[0m"  << endl;
          }
        } else {
          cout << "\033[33m" << endl; //unix only
          cerr << "warning: missing value for \"" << identifier << "\" at line " << lineNb << ", line ignored" << endl;
          cout << "\033[0m"  << endl;
        }
      }
      lineNb++;
    }
  }
}


bool ConfigReader::ParseConvergenceType(string arg, Settings& params) {
    params.ConvergenceType( arg );
    /*
  bool success = true;
  if (arg == "DP") {
      params.ConvergenceType(ConvergenceType::DP);
  } else if (arg == "DP_OMP") {
    params.SetConvergenceType(ConvergenceType::DP_OMP);
  } else if (arg == "DP_OMP_AVX") {
    params.SetConvergenceType(ConvergenceType::DP_OMP_AVX);
  } else if (arg == "DP_OMP_AVX+") {
    params.SetConvergenceType(ConvergenceType::DP_OMP_AVXPLUS);
  } else if (arg == "DP_OMP_AVX++") {
    params.SetConvergenceType(ConvergenceType::DP_OMP_AVXPLUSPLUS);
  } else if (arg == "SP") {
    params.SetConvergenceType(ConvergenceType::SP);
  } else if (arg == "SP_OMP") {
    params.SetConvergenceType(ConvergenceType::SP_OMP);
  } else if (arg == "SP_OMP_AVX") {
    params.SetConvergenceType(ConvergenceType::SP_OMP_AVX);
  } else if (arg == "SP_OMP_AVX+") {
    params.SetConvergenceType(ConvergenceType::SP_OMP_AVXPLUS);
  } else if (arg == "SP_OMP_AVX++") {
    params.SetConvergenceType(ConvergenceType::SP_OMP_AVXPLUSPLUS);
  } else if (arg == "FP") {
    params.SetConvergenceType(ConvergenceType::FP);
  } else if (arg == "FP_OMP") {
    params.SetConvergenceType(ConvergenceType::FP_OMP);
  } else if (arg == "FP_OMP") {
    params.SetConvergenceType(ConvergenceType::FP_OMP);
  } else if (arg == "FP_OMP_SSE2") {
    params.SetConvergenceType(ConvergenceType::FP_OMP_SSE2);
  } else if (arg == "FP_OMP_AVX2") {
    params.SetConvergenceType(ConvergenceType::FP_OMP_AVX2);
  } else if (arg == "CUDA_D") {
    params.SetConvergenceType(ConvergenceType::CUDA_D);
  } else if (arg == "CUDA_D2") {
    params.SetConvergenceType(ConvergenceType::CUDA_D2);
  } else if (arg == "CUDA_D2_WP") {
    params.SetConvergenceType(ConvergenceType::CUDA_D2_WP);
  } else if (arg == "CUDA_S") {
    params.SetConvergenceType(ConvergenceType::CUDA_S);
  } else {
    success = false;
  }*/
  return true;
}


bool ConfigReader::ParseColorMap(string arg, Settings& params) {
    params.ColorMapMode( arg );
    /*
  bool success = true;
  if (arg == "GREYSCALE") {
      params.SetColorMapMode(ColorMapMode::GREYSCALE);
  } else if (arg == "HSV") {
      params.SetColorMapMode(ColorMapMode::HSV);
  } else if (arg == "HSV_LOOP") {
      params.SetColorMapMode(ColorMapMode::HSV_LOOP);
  } else if (arg == "HSV_SHIFTED") {
      params.SetColorMapMode(ColorMapMode::HSV_SHIFTED);
  } else if (arg == "SMOOTH") {
      params.SetColorMapMode(ColorMapMode::SMOOTH);
  } else if (arg == "SMOOTH_LOOP") {
      params.SetColorMapMode(ColorMapMode::SMOOTH_LOOP);
  } else if (arg == "SMOOTH_SHIFTED") {
      params.SetColorMapMode(ColorMapMode::SMOOTH_SHIFTED);
  } else if (arg == "LERP_BLUE_RED") {
      params.SetColorMapMode(ColorMapMode::LERP_BLUE_RED);
  } else if (arg == "LSD") {
      params.SetColorMapMode(ColorMapMode::LSD);
  } else {
      success = false;
  }*/
  return true;
}


vector< vector<string> > CSVHandler::ReadCSV(string filename, Settings& params) {
  ifstream file;
  if (filename.length() <= 4 || filename.substr(filename.length()-4) != ".csv") {
    filename+= ".csv";
  }

  vector< vector<string> > tab;
  file.open(filename.c_str(), fstream::in | fstream::out);
  if (!file) {
    cout << "\033[31m" << endl; //unix only
    cerr << "error: can't open \"" << filename << "\", log disabled" << endl;
    cout << "\033[0m"  << endl;
    params.SetLogToFile(false);
  } else {
    int rowNb = 0;
    string line;
    while (std::getline(file, line)) {
      vector<string> row = splitString(line, ",");
      /*vector<string> row;
      string element;
      while (std::getline(line, element, ',')) {
        row.push_back(element);
      }*/
      tab.push_back(row);
    }
    return tab;
  }
  return tab;
}




bool CSVHandler::WriteCSV(vector< vector<string> > tab, string filename, Settings& params) {
  ofstream file;
  if (filename.length() <= 4 || filename.substr(filename.length()-4) != ".csv") {
    filename+= ".csv";
  }

  file.open(filename.c_str());
  if (!file) {
    cout << "\033[31m" << endl; //unix only
    cerr << "error: can't save to \"" << filename << "\", log disabled" << endl;
    cout << "\033[0m"  << endl;
    params.SetLogToFile(false);
    return false;
  } else {
    for (size_t i = 0 ; i < tab.size() ; i++) {
      for (size_t j = 0 ; j < tab[i].size() ; j++) {
        file << tab[i][j] << ",";
      }
      file << std::endl;
    }
  }
  return true;
}



void CSVHandler::PrintCSV(vector< vector<string> > tab) {
  for (size_t i = 0 ; i < tab.size() ; i++) {
    for (size_t j = 0 ; j < tab[i].size() ; j++) {
      std::cout << tab[i][j] << ",";
    }
    std::cout << std::endl;
  }
}
