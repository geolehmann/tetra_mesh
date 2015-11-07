#pragma once
#include <string>
#include <cuda_runtime.h>
#include <stdint.h>

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

float4& normalize(const float4 &a){ float f = sqrt(a.x*a.x + a.y*a.y + a.z*a.z); return make_float4(a.x/f,a.y/f,a.z/f,0); }


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

double intersect_dist(const Ray ray, float4 a, float4 b, float4 c) //tested and works!!
{
	float4 v0v1 = b - a;
	float4 v0v2 = c - a;
	float4 pvec = Cross(ray.d,v0v2);
	float det = Dot(v0v1,pvec);
	float invDet = 1 / det;
	float4 tvec = ray.o - a;
	float4 qvec = Cross(tvec,v0v1);
	return abs(Dot(v0v2,qvec) * invDet);
}


float4 camcr(double w, double h, const double x, const double y)
{
	// taken from smallpaint by karoly zsolnai
	float fovx = PI / 4;
	float fovy = (h / w) * fovx;
	return make_float4(((2*x-w)/w)* tanf(fovx),-((2*y-h)/h)*tanf(fovy),-1.0,0);
}

struct RGB
{
	double x, y, z;
	RGB(double x0, double y0, double z0){ x = x0; y = y0; z = z0; }
	RGB(double xyz0 = 0){ x = xyz0; y = xyz0; z = xyz0; }
};

