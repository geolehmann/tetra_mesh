#pragma once
#include <string>
#include <cuda_runtime.h>
#include <stdint.h>
#include <random>

#define PI 3.1415926536

struct Vec {
	double x, y, z;
	Vec(double x0, double y0, double z0){ x = x0; y = y0; z = z0; }
	Vec(double xyz0 = 0){ x = xyz0; y = xyz0; z = xyz0; }
	Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
	Vec operator+=(const Vec &b) { x += b.x; y += b.y; z += b.z; return (*this); }
	Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
	Vec operator*(double b) const { return Vec(x*b, y*b, z*b); }
	Vec operator*(const Vec &b) const { return Vec(x*b.x, y*b.y, z*b.z); }
	Vec operator*=(const Vec &b) { x *= b.x; y *= b.y; z *= b.z; return (*this); }
	Vec operator/(double b) const { return Vec(x / b, y / b, z / b); }
	Vec operator/(const Vec &b) const { return Vec(x / b.x, y / b.y, z / b.z); }
	bool operator<(const Vec &b) const { return x < b.x && y < b.y && z < b.z; }
	bool operator>(const Vec &b) const { return x > b.x && y > b.y && z > b.z; }
	Vec& norm(){ return *this = *this * (1 / sqrt(x*x + y*y + z*z)); }
	double length() const { return sqrt(x*x + y*y + z*z); }
	double dot(const Vec &b) const { return x*b.x + y*b.y + z*b.z; }
	double avg() const { return (x + y + z) / 3.0; }
	double max() const { return x > y ? (x > z ? x : z) : (y > z ? y : z); }
	double min() const { return x < y ? (x < z ? x : z) : (y < z ? y : z); }
	Vec operator%(const Vec &b) const { return Vec(y*b.z - z*b.y, z*b.x - x*b.z, x*b.y - y*b.x); }
	const double& operator[](size_t i) const { return i == 0 ? x : (i == 1 ? y : z); }
};


float4 operator-(const float4 &a, const float4 &b) {

	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0);

}

float4 operator+(const float4 &a, const float4 &b) {

	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0);

}

float4 operator*(const float4 &a, const double &b) {

	return make_float4(a.x*b, a.y*b, a.z*b, 0);

}



float4 normalize(const float4 &a)
{ 
	float f = sqrt(a.x*a.x + a.y*a.y + a.z*a.z); 
	return make_float4(a.x/f, a.y/f, a.z/f, 0);
}


inline float Dot(const float4 a, const float4 b)
{
	return  a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float4 Cross(const float4 a, const float4 b)
{
	float4 cross = { a.y * b.z - a.z * b.y,
					 a.z * b.x - a.x * b.z,
					 a.x * b.y - a.y * b.x };
	return cross;
}

struct Ray 
{ 
	float4 o, d; 
};

float ScTP(const float4 a, const float4 b, const float4 c)
{
	return Dot(a, Cross(b, c));
}

int signf(float x)
{
	if (x > 0.f) return 1;
	if (x < 0.f) return -1;
	return 0;
}


bool SameSide(float4 v1, float4 v2, float4 v3, float4 v4, float4 p)
{
	float4 normal = Cross(v2 - v1, v3 - v1);
	float dotV4 = Dot(normal, v4 - v1);
	float dotP = Dot(normal, p - v1);
	return signf(dotV4) == signf(dotP);
}


struct BBox
{
	float4 min, max;
};


float4 getNormal(float4 a, float4 b, float4 c) {	return(Cross(b-a,c-a)); }

double intersect_dist(Ray ray, float4 a, float4 b, float4 c) //tested and works!!
{
	float4 N = normalize(Cross(b - a, c - a));
	float D = Dot(N, a) * -1;
	float NdotR = Dot(N, ray.d)* -1;
	return (Dot(N, ray.o) + D) / NdotR;
}


struct RGB
{
	double x, y, z;
	RGB(double x0, double y0, double z0){ x = x0; y = y0; z = z0; }
	RGB(double xyz0 = 0){ x = xyz0; y = xyz0; z = xyz0; }
	RGB operator/(double b) const { return RGB(x / b, y / b, z / b); }
	RGB operator+(const RGB &b) const { return RGB(x + b.x, y + b.y, z + b.z); }
};

// Random Number Generation, from karoly zsolnai
std::mt19937 mersenneTwister;
std::uniform_real_distribution<double> uniform;
#define RND (2.0*uniform(mersenneTwister)-1.0)	