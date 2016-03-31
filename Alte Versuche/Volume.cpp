#include "Volume.h"



__managed__ Sphere* d_homogeneousMedium;




		/* ------------------------------------------------------------------- ---------------------------------------------------- */
		/* ---------------------------------------------volumetric pathtracing ---------------------------------------------------- */
		/* ------------------------------------------------------------------- ---------------------------------------------------- */

		float t_s = dist, t_m = 1e7f, tnear_m, tfar_m;
		float4 absorption = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
		bool intrsctmd = (t_m = intersect_sphere(d_homogeneousMedium, &Ray(originInWorldSpace, rayInWorldSpace), &tnear_m, &tfar_m)) > 0.0f; // intersection with medium?
		//bool intrscts = intersect_all_spheres(&r, d_spheres, nbSpheres, t_s, id); // intersection with spheres?
		bool intrscts = true;
		//if (!intrscts && !intrsctmd)
		//	break; // <- at least tets always hit

		bool doAtmosphericScattering = (intrsctmd && (!intrscts || (t_m <= t_s || (t_s >= tnear_m && t_s <= tfar_m))));

		// hier evtl fehler
		if (intrscts && (firsthit.refl_t == REFR || (firsthit.refl_t & VOL) == VOL) && Dot(originInWorldSpace + rayInWorldSpace * t_s - d_homogeneousMedium->p, rayInWorldSpace) >= 0.0f) doAtmosphericScattering = false;
		//Sphere *obj = &d_spheres[id];
		float t = t_s;
		if (doAtmosphericScattering) {
			//obj = d_homogeneousMedium;
			firsthit.refl_t = VOL;
			t = t_m;
		}

		if ((firsthit.refl_t & VOL) == VOL && Dot(originInWorldSpace + rayInWorldSpace * t - d_homogeneousMedium->p, rayInWorldSpace) >= 0.0f) {
			Ray sRay(make_float4(0, 0, 0, 0), make_float4(0, 0, 0, 0));
			float e0 = curand_uniform(randState), e1 = curand_uniform(randState), e2 = curand_uniform(randState);

			float s, ms = scatter(Ray(originInWorldSpace, rayInWorldSpace), sRay, d_homogeneousMedium->sigma_s, s, e0, e1, e2);
			float distToExit = t_s < t ? t_s : t;
			if (s <= distToExit && d_homogeneousMedium->sigma_s > 0) {
				originInWorldSpace = sRay.o;
				rayInWorldSpace = sRay.d;
				mask *= (d_homogeneousMedium->scatteringColor * ms);
				absorption = make_float4(1.0f, 1.0f, 1.0f, 0) + d_homogeneousMedium->absorptionColor * (expf(-d_homogeneousMedium->sigma_a * s) - 1.0f);
				mask *= absorption;
				continue;
			}
			float dist_ = t_m;
			// Ray is probably leaving the medium
			if (intrscts && t_s <= t) {
				//obj = &d_spheres[id];
				t = t_s;
				dist_ = t_s;
			}
			absorption = make_float4(1.0f, 1.0f, 1.0f, 0) + d_homogeneousMedium->absorptionColor * (expf(-d_homogeneousMedium->sigma_t * dist_) - 1.0f);
		}

		/* ------------------------------------------------------------------- ---------------------------------------------------- */
		/* ---------------------------------------------volumetric pathtracing ---------------------------------------------------- */
		/* ------------------------------------------------------------------- ---------------------------------------------------- */




	// ===========================
	//     sphere for medium
	// ===========================
	gpuErrchk(cudaMallocManaged(&d_homogeneousMedium, sizeof(Sphere)));
	d_homogeneousMedium->rad = 200.0f;
	d_homogeneousMedium->p = make_float4(0.0f, 0.0f, 0.0f, 0);
	d_homogeneousMedium->e = make_float4(0, 0, 0, 0);
	d_homogeneousMedium->c = make_float4(0, 0, 0, 0);
	d_homogeneousMedium->refl = VOL;
	d_homogeneousMedium->absorptionColor = make_float4(1.0f, 1.0f, 1.0f, 0);
	d_homogeneousMedium->scatteringColor = make_float4(0.8f, 0.8f, 0.8f, 0);
	d_homogeneousMedium->sigma_s = 0.08f;
	d_homogeneousMedium->sigma_a = 0.001f;



