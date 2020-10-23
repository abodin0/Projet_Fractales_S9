/********************
       Utils
********************/

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <vector>
//#include <bits/stdc++.h>
#include <sys/utsname.h>
#include <time.h>
#include <string>
#include <fstream>
#include <sstream>


using namespace std;


int map(int val, int inMin, int inMax, int outMin, int outMax);
unsigned int median(std::vector<unsigned int> values);
string dateTimeString();

string getCPUname();
string getOSname();

#endif /*UTILS_HPP_*/
