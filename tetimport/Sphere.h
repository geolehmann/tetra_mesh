#pragma once
#include "Math.h"

__device__ float sphIntersect( float4 ro, float4 rd, float4 sph, float rad )
{
	// taken from iq - sphere density
	float4 oc = ro - sph;
	float b = Dot( oc, rd );
	float c = Dot( oc, oc ) - rad*rad;
	float h = b*b - c;
	if( h<0.0 ) return -1.0;
    h = sqrt( h );
	return -b - h;
}
