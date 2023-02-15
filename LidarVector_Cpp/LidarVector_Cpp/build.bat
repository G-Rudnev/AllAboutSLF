g++ -shared LidarVector_Cpp.cpp Line.cpp ^
-o ..\\pyRobo\\lidarVector.pyd ^
-std=c++17 ^
-IC:\\Python\\Python38\\Lib\\site-packages\\numpy\\core\\include ^
-IC:\\Python\\Python38\\include ^
-LC:\\Python\\Python38\\libs ^
-LC:\\Python\\Python38\\Lib\\site-packages\\numpy\\core\\lib ^
-lpython38 ^
-O3