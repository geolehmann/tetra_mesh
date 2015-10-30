#include "tetgen_io.h"

#define NUM_BLOCKS    256
#define NUM_THREADS   256

class rayhit
{
public:
	int32_t tet;
	int32_t face;
	float4 pos;
	bool wall=false;
	bool constrained=false;
};



void traverse_ray(tetrahedra_mesh *mesh, int32_t start, float t, rayhit *d)
{
	int32_t nexttet, nextface, lastface = 0;
	while (1)
	{
		uVec2 hit;
		GetExitTet(mesh->cam.o,mesh->cam.d, mesh->get_tetrahedra(start), lastface, nextface,nexttet);
		fprintf(stderr, "Number of next tetrahedra: %lu \n", nexttet);
		fprintf(stderr, "Number of next face: %lu \n\n", nextface);

		if (mesh->get_face(nexttet.face).face_is_constrained == true) { d->constrained = true; hit.face = nexttet.face; hit.tet = nexttet.tet; break; }
		if (mesh->get_face(nexttet.face).face_is_wall == true) { d->wall = true; hit.face = nexttet.face; hit.tet = nexttet.tet; break; }
		if (nexttet.tet == -1) { d->wall = true; hit.face = nexttet.face; hit.tet = nexttet.tet; break; } // when adjacent tetrahedra is -1, ray stops
		lastface = nexttet.face;
		start = nexttet.tet;
	}
	if (d->wall == true) fprintf_s(stderr, "Wall hit.\n"); // finally... (27.10.2015)
	if (d->constrained == true) fprintf_s(stderr, "Triangle hit.\n"); // now i have: index of face(=triangle) which is intersected by ray.
}



int main()
{
	const float t = 1.0e+30f;
	// load tetrahedral model
	tetrahedra_mesh tetmesh;
	tetmesh.load_tet_ele("untitled.1.ele");
	tetmesh.load_tet_neigh("untitled.1.neigh");
	tetmesh.load_tet_node("untitled.1.node");
	tetmesh.load_tet_face("untitled.1.face");
	tetmesh.load_tet_t2f("untitled.1.t2f");
	tetmesh.cam.d = make_float4(0, 1, 0, 0);
	tetmesh.cam.o = make_float4(0, 5, 5, 0);

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


	rayhit firsthit;
	traverse_ray(&tetmesh, start, t, &firsthit);


	system("PAUSE");
}


