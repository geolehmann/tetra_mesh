/*
*  tetrahedra-based raytracer
*  Copyright (C) 2015  Christian Lehmann
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*/

#pragma once
#include <string>
#include <cuda_runtime.h>
#include <random>

#define PI 3.1415926535897932384626422832795028841971f
#define TWO_PI 6.2831853071795864769252867665590057683943f
#define PI_OVER_TWO 1.5707963267948966192313216916397514420985f
#define epsilon 1e-8
#define inf 1e20

typedef int int32_t;
typedef unsigned int uint32_t;

enum Refl_t { DIFF, SPEC, REFR, VOL, METAL}; 
enum Geometry { TRIANGLE, SPHERE };

// ----------------  CUDA float operations -------------------------------

inline __host__ __device__ float3 operator/(const float3 &a, const int &b) { return make_float3(a.x / b, a.y / b, a.z / b); }
inline __host__ __device__ float3 operator+(const float3 &a, const float3 &b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __host__ __device__ float3 operator+=(float3 &a, const float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return make_float3(0, 0, 0); }

inline __host__ __device__ float4 operator-=(float4 &a, const float4 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;	return make_float4(0, 0, 0, 0); }
inline __host__ __device__ float4 operator+(const float4 &a, const float4 &b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0); }
inline __host__ __device__ float4 operator-(const float4 &a, const float4 &b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0); }
inline __host__ __device__ float4 operator*(float4 &a, float4 &b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, 0); }
inline __host__ __device__ float4 operator*(const float4 &a, const float &b) { return make_float4(a.x*b, a.y*b, a.z*b, 0); }
inline __host__ __device__ float4 operator*(const float &b, const float4 &a) { return make_float4(a.x * b, a.y * b, a.z * b, 0); }
inline __host__ __device__ void operator*=(float4 &a, float4 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; }
inline __host__ __device__ void operator*=(float4 &a, float &b) { a.x *= b; a.y *= b; a.z *= b; }
inline __host__ __device__ float4 operator/(const float4 &a, const float &b) { return make_float4(a.x / b, a.y / b, a.z / b, 0); }
inline __host__ __device__ float4 operator+=(float4 &a, const float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; return make_float4(0, 0, 0, 0); }

// ------------------------CUDA math --------------------------------------------------

__device__ float4 normalize(float4 &a)
{ 
	float f = 1/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z); 
	return make_float4(a.x*f, a.y*f, a.z*f, 0);
}

 __device__  __host__ float Dot(const float4 &a, const float4 &b)
{
	return  a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float4 reflect(const float4 &i,const float4 &n)
{
	return i - 2.0f * n * Dot(n, i);
}

__device__  float4 Cross(const float4 &a, const float4 &b)
{
	return make_float4( a.y * b.z - a.z * b.y, 
						a.z * b.x - a.x * b.z, 
						a.x * b.y - a.y * b.x, 0);
}

__device__ float ScTP(const float4 &a, const float4 &b, const float4 &c)
{
	// computes scalar triple product
	return Dot(a, Cross(b, c));
}

__device__ int signf(float f) 
{
	if (f > 0.0) return 1;
	if (f < 0.0) return -1;
	return 0;
}

__device__ bool SameSide(const float4 &v1, const float4 &v2, const float4 &v3, const float4 &v4, const float4 &p)
{
	float4 normal = Cross(v2 - v1, v3 - v1);
	float dotV4 = Dot(normal, v4 - v1);
	float dotP = Dot(normal, p - v1);
	return signf(dotV4) == signf(dotP);
}

inline __device__ __host__ float clamp(float f, float a, float b)
{
	return fmaxf(a, fminf(f, b));
}

__device__ bool HasNaNs(float4 a) { return isnan(a.x) || isnan(a.y) || isnan(a.z); }

struct RGB
{
	float x, y, z;
	__device__ RGB(float x0, float y0, float z0) { x = x0; y = y0; z = z0; }
	__device__ RGB(float xyz0){ x = xyz0; y = xyz0; z = xyz0; }
	__device__ RGB operator/(const float &b) const { return RGB(x / b, y / b, z / b); }
	__device__ RGB operator+(const RGB &b) const { return RGB(x + b.x, y + b.y, z + b.z); }
	__device__ RGB operator*(const float &b) const { return RGB(x * b, y * b, z * b); }
};
__device__ RGB operator+=(RGB &a, const RGB b) { a.x += b.x; a.y += b.y; a.z += b.z; return RGB(a.x,a.y,a.z); }
__device__ float3 operator+=(float3 &a, const RGB b) { a = make_float3(a.x + b.x, a.y + b.y, a.z + b.z); return a; }

__device__ RGB de_nan(RGB &a)
{
	// from http://psgraphics.blogspot.de/2016/04/debugging-by-sweeping-under-rug.html
	RGB temp = a;
	if (!(temp.x == temp.x)) temp.x = 0;
	if (!(temp.y == temp.y)) temp.y = 0;
	if (!(temp.z == temp.z)) temp.z = 0;
	temp.z = 0;
	return temp;
}

struct Ray
{
	float4 o, d;
	__device__ Ray(float4 o_, float4 d_) : o(o_), d(d_) {}
};

__device__ bool nearlyzero(float a)
{
	if (a > 0.0 && a < 0.01) return true;
	return false;
}

__device__ float intersect_dist(const Ray &ray, const float4 &a, const float4 &b, const float4 &c, bool &isEdge)
{
	float4 pq = ray.d;
	float4 pa = a - ray.o;
	float4 pb = b - ray.o;
	float4 pc = c - ray.o;

	float u = ScTP(pq, pc, pb);
	float v = ScTP(pq, pa, pc);
	float w = ScTP(pq, pb, pa);
	float denom = 1.0f / (u + v + w);
	u *= denom;
	v *= denom;
	w *= denom;

	if (nearlyzero(u) || nearlyzero(w) || nearlyzero(v)) isEdge = true;

	float4 e1 = b - a;
	float4 e2 = c - a;
	float4 q = Cross(ray.d, e2);
	float d = Dot(e1, q);
	float4 s = ray.o - a;
	float4 r = Cross(s, e1);
	return Dot(e2, r) / d;

	/*float4 e1 = b - a;
	float4 e2 = c - a;
	float4 N = Cross(e1, e2);
	double NdotRayDirection = Dot(N,ray.d); 
	double d = Dot(N,a); 
	return (Dot(N,ray.o) + d) / NdotRayDirection; */

}

__device__ float4 getTriangleNormal(const float4 &p1, const float4 &p2, const float4 &p3)
{
	return(Cross(p2 - p1, p3 - p1));
}

// ----------------------- non-CUDA math -----------------------

float4 normalizeCPU(const float4 &a)
{
	float f = 1/sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
	return make_float4(a.x * f, a.y * f, a.z * f, 0);
}

float4 CrossCPU(const float4 &a, const float4 &b)
{
	float4 cross = { a.y * b.z - a.z * b.y,
					a.z * b.x - a.x * b.z,
					a.x * b.y - a.y * b.x };
	return cross;
}

float DotCPU(const float4 &a, const float4 &b)
{
	return  a.x * b.x + a.y * b.y + a.z * b.z;
}

int signfCPU(float f) 
{
	if (f > 0.0) return 1;
	if (f < 0.0) return -1;
	return 0;
}

bool SameSideCPU(const float4 &v1, const float4 &v2, const float4 &v3, const float4 &v4, const float4 &p)
{
	float4 normal = CrossCPU(v2 - v1, v3 - v1);
	float dotV4 = DotCPU(normal, v4 - v1);
	float dotP = DotCPU(normal, p - v1);
	return signfCPU(dotV4) == signfCPU(dotP);
}

// ------------------------------- structure definitions -----------------------------

struct BBox
{
	float4 min, max;
};

struct mesh2
{
	// nodes
	uint32_t *n_index;
	float *n_x, *n_y, *n_z;

	//faces
	uint32_t *f_index;
	uint32_t *f_node_a, *f_node_b, *f_node_c;
	bool *face_is_constrained = false;
	bool *face_is_wall = false;

	// tetrahedra
	uint32_t *t_index;
	int32_t *t_findex1, *t_findex2, *t_findex3, *t_findex4;
	int32_t *t_nindex1, *t_nindex2, *t_nindex3, *t_nindex4;
	int32_t *t_adjtet1, *t_adjtet2, *t_adjtet3, *t_adjtet4;

	//mesh 
	uint32_t tetnum, nodenum, facenum, edgenum;
};

struct rayhit
{
	float4 pos;
	float4 color;
	float4 ref;
	Refl_t refl_t;
	int32_t tet = 0;
	int32_t face = 0;
	int depth = 0;
	bool wall = false;
	bool constrained = false;
	bool dark = false; // if hit is too far away
};