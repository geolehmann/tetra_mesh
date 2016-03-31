#pragma once
#include "Math.h"

struct Sphere {
	float rad; // radius
	float4 p, e, c; // position, emission, color
	Refl_t refl; // reflection type (DIFFuse, SPECular, REFRactive)
	float sigma_s, sigma_a, sigma_t; // scattering, absorption and extinction coefficients
	float4 scatteringColor, absorptionColor; // scattering, absorption
	float ior;

	//	Sphere(float rad_, float4 p_, float4 e_, float4 c_, Refl_t refl_, float4 scatteringColor_, float4 absorptionColor_, float sigma_s_, float sigma_a_, float _ior = 1.5f) : rad(rad_), p(p_), e(e_), c(c_),
	//	refl(refl_), scatteringColor(scatteringColor_), absorptionColor(absorptionColor_), sigma_s(sigma_s_), sigma_a(sigma_a_), ior(_ior) {}

};

__device__ float intersect_sphere(const Sphere *s, const Ray *r, float *tin, float *tout) 
{
	float4 op = s->p - r->o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
	float t, eps = 1e-3f, b = Dot(op, r->d), det = b * b - Dot(op, op) + s->rad*s->rad;
	if (det < 0.0f) return 0.0f;
	else
		det = sqrtf(det);
	if (tin && tout) {
		*tin = (b - det <= 0.0f) ? 0.0f : b - det;
		*tout = b + det;
	}
	return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0.0f);
}

__device__ bool intersect_all_spheres(const Ray *r, const Sphere *d_spheres, int nbSpheres, float &t, int &id) {
	float d, infi = t = 1e7f, tnear, tfar;
	for (int i = nbSpheres; i--;)
	if ((d = intersect_sphere(&d_spheres[i], r, &tnear, &tfar)) && (d < t)) {
		t = d;
		id = i;
	}
	return t < infi;
}

__device__ inline float sampleSegment(float epsilon1, float sigma) {
	return -logf(1.0f - epsilon1) / sigma;
}
__device__ inline float4 sampleSphere(float e1, float e2) {
	float z = 1.0f - 2.0f * e1, sint = sqrtf(1.0f - z * z);
	return make_float4(cosf(2.0f * PI * e2) * sint, sinf(2.0f * PI * e2) * sint, z, 0.0f);
}
__device__ inline float4 sampleHG(float g, float e1, float e2) {
	//float s=2.0*e1-1.0, f = (1.0-g*g)/(1.0+g*s), cost = 0.5*(1.0/g)*(1.0+g*g-f*f), sint = sqrtf(1.0-cost*cost);
	float s = 1.0f - 2.0f*e1, denom = 1.0f + g*s;
	float cost = (s + 2.0f*g*g*g*(-1.0f + e1) * e1 + g*g*s + 2.0f*g*(1.0f - e1 + e1*e1)) / (denom * denom), sint = sqrtf(1.0f - cost*cost);
	return make_float4(cosf(2.0f * PI * e2) * sint, sinf(2.0f * PI * e2) * sint, cost, 0.0f);
}
__device__  void generateOrthoBasis(float4 &u, float4 &v, float4 w) {
	float4 coVec;
	if (fabs(w.x) <= fabs(w.y))
	if (fabs(w.x) <= fabs(w.z)) coVec = make_float4(0, -w.z, w.y, 0.0f);
	else coVec = make_float4(-w.y, w.x, 0, 0.0f);
	else if (fabs(w.y) <= fabs(w.z)) coVec = make_float4(-w.z, 0, w.x, 0.0f);
	else coVec = make_float4(-w.y, w.x, 0, 0.0f);
	coVec = normalize(coVec);
	u = Cross(w, coVec),
		v = Cross(w,u);
}
__device__ inline float scatter(const Ray &r, Ray &sRay, float sigma_s, float &s, float e0, float e1, float e2) {
	s = sampleSegment(e0, sigma_s);
	float4 x = r.o + r.d * s;
	//Vec dir = sampleSphere(e1, e2); //Sample a direction ~ uniform phase function
	float4 dir = sampleHG(-0.0f, e1, e2); //Sample a direction ~ Henyey-Greenstein's phase function
	float4 u, v;
	generateOrthoBasis(u, v, r.d);
	dir = u * dir.x + v * dir.y + r.d * dir.z;
	sRay = Ray(x, dir);
	return 1.0f;
}


