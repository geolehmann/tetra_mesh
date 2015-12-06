#pragma once
#include <string>
#include <cuda_runtime.h>
#include <stdint.h>
#include <random>

#define PI 3.1415926536

__device__ float4 operator-(const float4 &a, const float4 &b) {

	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0);

}

__device__ float4 operator+(const float4 &a, const float4 &b) {

	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0);

}

__device__ float4 operator*(const float4 &a, const float &b) {

	return make_float4(a.x*b, a.y*b, a.z*b, 0);

}

__device__ float4 operator/(const float4 &a, const float &b) {

	return make_float4(a.x/b, a.y/b, a.z/b, 0);

}

__device__ float4 normalize(float4 a)
{ 
	float f = sqrt(a.x*a.x + a.y*a.y + a.z*a.z); 
	return make_float4(a.x/f, a.y/f, a.z/f, 0);
}


__device__  float Dot(const float4 a, const float4 b)
{
	return  a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__  float4 Cross(const float4 a, const float4 b)
{
	float4 cross = { a.y * b.z - a.z * b.y,
					 a.z * b.x - a.x * b.z,
					 a.x * b.y - a.y * b.x };
	return cross;
}

struct Ray 
{ 
	float4 o, d, u; 
};

__device__ float ScTP(const float4 a, const float4 b, const float4 c)
{
	return Dot(a, Cross(b, c));
}

__device__ int signf(float x)
{
	if (x >= 0.f) return 1;
	if (x < 0.f) return -1;
	return 0;
}


__device__ bool SameSide(float4 v1, float4 v2, float4 v3, float4 v4, float4 p)
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


__device__ float4 getNormal(float4 a, float4 b, float4 c)
{	
	return(Cross(b-a,c-a)); 
}

struct RGB
{
	float x, y, z;
	__device__ RGB(float x0, float y0, float z0){ x = x0; y = y0; z = z0; }
	__device__ RGB(float xyz0 = 0){ x = xyz0; y = xyz0; z = xyz0; }
	__device__ RGB operator/(float b) const { return RGB(x / b, y / b, z / b); }
	__device__ RGB operator+(const RGB &b) const { return RGB(x + b.x, y + b.y, z + b.z); }
};

__device__ float intersect_dist(Ray ray, float4 a, float4 b, float4 c) //tested and works!!
{
float4 e1 = b - a;
float4 e2 = c - a;
float4 q = Cross(ray.d,e2);
float a_ = Dot(e1,q);
float4 s = ray.o - a;
float4 r = Cross(s,e1);
return Dot(e2,r) / a_;
}

// from rayito - github.com/Tecla/Rayito
__device__ Ray makeCameraRay(float fieldOfViewInDegrees, const float4& origin, const float4& target, const float4& targetUpDirection, float xScreenPos0To1, float yScreenPos0To1)
{
	float4 forward = normalize(target - origin);
	float4 right = normalize(Cross(forward, targetUpDirection));
	float4 up = normalize(Cross(right, forward));
	float tanFov = std::tan(fieldOfViewInDegrees * PI / 180.0f);
	Ray ray;
	ray.o = origin;
	ray.d = forward + right * ((xScreenPos0To1 - 0.5f) * tanFov) + up * ((yScreenPos0To1 - 0.5f) * tanFov);
	ray.d = normalize(ray.d);
	return ray;
}

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


inline __device__ __host__ float clamp(float f, float a, float b)
{
	return fmaxf(a, fminf(f, b));
}