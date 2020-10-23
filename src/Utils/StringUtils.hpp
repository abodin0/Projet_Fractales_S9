/*

*/

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>

#ifndef STRINGUTILS_H_
#define STRINGUTILS_H_

using namespace std;

void removeAllCharFromString(char c, string* str);
void removeAllCharFromStringExcept(char c, string* str);
vector<string> splitString(string str, string delimiter);
vector<string> splitStringExcept(string str, string delimiter);
void stringToUppercase(string* str);
int stringToInt(string s);
float stringToFloat(string s);
double stringToDouble(string s);
bool stringEquals(string str, string content);
bool stringContains(string str, string content);

template <typename T> std::string ToString(T toString) {
  ostringstream ss;
  ss << toString;
  return ss.str();
}

#endif /*STRINGUTILS_H_*/
