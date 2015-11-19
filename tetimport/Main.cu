#define NO_MSG //increases speed greatly
#include "tetgen_io.h"

const int width = 320, height=240;


RGB trace(Ray r, tetrahedra_mesh *mesh, int32_t start, int depth)
{
	rayhit firsthit;
	traverse_ray(mesh, r, start, firsthit, depth);

	face fc = mesh->get_face(firsthit.face);
	node a1 = mesh->get_node(fc.node_a);
	node a2 = mesh->get_node(fc.node_b);
	node a3 = mesh->get_node(fc.node_c);
	float c = abs(intersect_dist(r, a1.f_node(), a2.f_node(), a3.f_node()));
	
	float k = ((255-0) / (0-10)); // in zweiter klammer erster wert ist untere grenze distanzwerte
	float d = 0 - (10 * k);
	float d2 = (c*k) + d;

	return RGB(c,fc.index,d2); // return depth value
}


// from rayito
Ray makeCameraRay(float fieldOfViewInDegrees,
	const float4& origin,
	const float4& target,
	const float4& targetUpDirection,
	float xScreenPos0To1,
	float yScreenPos0To1)
{
	float4 forward = normalize(target - origin);
	float4 right = normalize(Cross(forward, targetUpDirection));
	float4 up = normalize(Cross(right, forward));

	// Convert to radians, as that is what the math calls expect
	float tanFov = std::tan(fieldOfViewInDegrees * PI / 180.0f);

	Ray ray;

	// Set up ray info
	ray.o = origin;
	ray.d = forward +
		right * ((xScreenPos0To1 - 0.5f) * tanFov) +
		up * ((yScreenPos0To1 - 0.5f) * tanFov);
	ray.d = normalize(ray.d);

	return ray;
}



int main()
{

	/*Ray test;
	test.d = make_float4(0, 0.3, 1, 0);
	test.o = make_float4(0.2, 4.3, -3, 0);

	double g = intersect_dist(test, make_float4(-2.2489, 2.0505, 0.3959, 0), make_float4(-2.2489, 9.9156, 0.3959, 0), make_float4(5.6161, 2.0505, 0.3959, 0));
	*/
	// nur ein einziges face von camera erfasst - kein wunder, dass alle dist gleich..........

	// load tetrahedral model
	tetrahedra_mesh tetmesh;
	tetmesh.load_tet_ele("untitled.1.ele");
	tetmesh.load_tet_neigh("untitled.1.neigh");
	tetmesh.load_tet_node("untitled.1.node");
	tetmesh.load_tet_face("untitled.1.face");
	tetmesh.load_tet_t2f("untitled.1.t2f");

	tetmesh.cam.o = make_float4(0, 7, 5, 0);
	tetmesh.cam.d = make_float4(1, 0, 0, 0);
	tetmesh.cam.u = make_float4(0, 0, 1, 0);

	// Get bounding box
	tetmesh.init_BBox();
	fprintf_s(stderr, "\nBounding box:MIN xyz - %f %f %f \n", tetmesh.boundingbox.min.x, tetmesh.boundingbox.min.y, tetmesh.boundingbox.min.z);
	fprintf_s(stderr, "             MAX xyz - %f %f %f \n\n", tetmesh.boundingbox.max.x, tetmesh.boundingbox.max.y, tetmesh.boundingbox.max.z);

	// Find camera tetrahedra
	int32_t start;
	start = tetmesh.GetTetrahedraFromPoint(tetmesh.cam.o);
	if (start == -1) {
		fprintf_s(stderr, "Camera point not inside mesh! Aborting\n");
		system("PAUSE");
		return 0;
	}	else fprintf_s(stderr, "Starting point (camera) tetra number: %lu\n\n",start);


	// raytracing stuff
	FILE *f = fopen("image.ppm", "w");
	FILE *f2 = fopen("test3.txt", "w");
	FILE *f3 = fopen("camera.txt", "w");
	fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
	
	#pragma omp parallel for
	for (int y = 0; y < height; ++y){
		float yu = 1.0f - (float(y) / float(height - 1));
		for (int x = 0; x < width; ++x){

				float xu = float(x) / float(width - 1);
				Ray ray = makeCameraRay(45.0f, tetmesh.cam.o, tetmesh.cam.d, tetmesh.cam.u, xu, yu);

				RGB c = trace(ray, &tetmesh, start, 0);

				if (y == 0 && x == 0) fprintf(f3, "0,0: %f \n", c.y);
				if (y == 0 && x == width - 1) fprintf(f3, "0, width: %f \n", c.y);
				if (y == height - 1 && x == 0) fprintf(f3, "height,0: %f \n", c.y);
				if (y == height-1 && x == width - 1) fprintf(f3, "height,width: %f \n", c.y);

				fprintf(f2, "%f \n", c.x);
				fprintf(f, "%d %d %d ", 0, 0, (int)c.z);
		}
	}
}


