/**********************\
      String Utils
***********************

Author :
--------
Jonathan SAUSSEREAU

Description :
-------------
Usefull functions to make it easier to handle strings

Changelog :
-----------
  - 2019/04/17 : Initial release
  - 2019/04/27 : Improved to hanle complex parsing
  - 2019/10/21 : Adapted to fit fractal application
*/

#include "StringUtils.hpp"

/*****************
      Utils
*****************/

void removeAllCharFromString(char c, string* str) {
  str->erase(std::remove(str->begin(), str->end(), c), str->end());
}

void removeAllCharFromStringExcept(char c, string* str) {
  vector<string> parts = splitStringExcept(*str, ";");
  stringstream res;
  copy(parts.begin(), parts.end(), ostream_iterator<string>(res, ""));
  *str = res.str();
}

vector<string> splitString(string str, string delimiter) {
  vector<string> parts;
  size_t pos = 0, lastPos = 0;
  do {
      pos = str.find(delimiter, lastPos);
      if (pos == string::npos) {
        pos = str.length();
      }
      string token = str.substr(lastPos, pos - lastPos);
      if (!token.empty()) {
        parts.push_back(token);
      } else {
        parts.push_back("");
      }
      //lastPos = pos + 1;
      lastPos = pos + delimiter.length();
  } while (pos < str.length() && lastPos < str.length());
  return parts;
}

vector<string> splitStringExcept(string str, string delimiter) {
  vector<string> parts;
  vector<bool> insideBrackets;
  for (unsigned long i = 0 ; i <= str.length() ; i++) {
    insideBrackets.push_back(false);
  }
  unsigned long pos = 0, lastPos = 0, lastRealPos = 0;
  unsigned long posEnd = 0, lastPosEnd = 0;

  // Scan str to exclude the string between '{' and '}' from being splitted
  do {
    pos = str.find("{", lastPos);
    posEnd = str.find("}", lastPosEnd);
    if (pos == string::npos) {
      pos = str.length();
    }
    if (posEnd == string::npos) {
      posEnd = str.length();
    }
    for (unsigned long i = pos ; i < posEnd ; i++) {
      insideBrackets.at(i) = true;
    }
    lastPos = pos + delimiter.length();
    lastPosEnd = posEnd + delimiter.length();
  } while (pos < str.length() && lastPos < str.length());

  pos = 0;
  lastPos = 0;
  do {
    pos = str.find(delimiter, lastRealPos);
    if (pos == string::npos) {
      pos = str.length();
    }
    if (insideBrackets.at(pos) != true) {
      string token = str.substr(lastPos, pos - lastPos);
      if (!token.empty()) {
        parts.push_back(token);
      }
      lastPos = pos + delimiter.length();
    }
    lastRealPos = pos + delimiter.length();
  } while (pos < str.length() && lastPos < str.length());
  return parts;
}

void stringToUppercase(string* str) {
  transform(str->begin(), str->end(), str->begin(), ::toupper);
}

void stringToLowercase(string* str) {
  transform(str->begin(), str->end(), str->begin(), ::tolower);
}

int stringToInt(string s) {
  return atoi(s.c_str());
}

float stringToFloat(string s) {
  return atof(s.c_str());
}

double stringToDouble(string s) {
  return atof(s.c_str());
}

bool stringContains(string str, string content) {
  return (str.find(content) != std::string::npos);
}

bool stringEquals(string str, string content) {
  return (str.compare(content) == 0);
}
