/********************
        Utils
********************/

#include "Utils.hpp"


int map(int val, int inMin, int inMax, int outMin, int outMax) {
  return (val - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}

unsigned int median(std::vector<unsigned int> values) {
  unsigned int vecsize = values.size();
  sort(values.begin(), values.end());

  unsigned int result;
  if (vecsize % 2 == 0) { // pair
    result = (values[vecsize/2 - 1] + values[vecsize/2]) / 2;
  } else { // impair
    result = values[vecsize/2];
  }
  return result;
}



string dateTimeString() {
  time_t now = time(0); //Number of seconds since January 1st,1970
  tm *ltm;
  ltm = localtime(&now);

  char buff[16];
  snprintf(buff, sizeof(buff), "%4d%02d%02d_%02d%02d%02d",
    (1900 + ltm->tm_year), (1 + ltm->tm_mon), (ltm->tm_mday),
    (ltm->tm_hour), (ltm->tm_min), (ltm->tm_sec));
  string out = buff;
  return out;
}

string getCPUname() {
  string line;
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (!cpuinfo) return "failed to get CPU info";
  while (getline(cpuinfo, line)) {
    stringstream str(line);
    string identifier;
    string info;
    if (getline(str, identifier, ':') && getline(str, info) && identifier.substr(0, 10) == "model name") {
      //cout << info << endl;
      return info;
    }
  }
  return "failed to get CPU info";
}

string getOSname() {
  struct utsname name;
  if (uname(&name)) {
    return "failed to get OS info";
  }
  string sysname = name.sysname; // linux
  string release = name.release; // version kernel
  string version = name.version; // distrib et version distrib
  string info = sysname + "\t" + release + "\t" + version ;
  //std::cout << info << std::endl;
  return info;
}
