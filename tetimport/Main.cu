#include "tetgen_io.h"

class rayhit
{
public:
	int32_t tet;
	int32_t face;
	float4 pos;
	bool wall=false;
	bool constrained=false;
};

void traverse_ray(tetrahedra_mesh *mesh, int32_t start, rayhit &d)
{
	int32_t nexttet, nextface, lastface = 0;

		while (1)
	{
		tetrahedra *h = &mesh->get_tetrahedra(start);
		int32_t findex[4] = { h->findex1, h->findex2, h->findex3, h->findex4 };
		int32_t adjtets[4] = { h->adjtet1, h->adjtet2, h->adjtet3, h->adjtet4 };
		float4 nodes[4] = { 
			make_float4(mesh->get_node(h->nindex1 - 1).x, mesh->get_node(h->nindex1 - 1).y, mesh->get_node(h->nindex1 - 1).z, 0),
			make_float4(mesh->get_node(h->nindex2 - 1).x, mesh->get_node(h->nindex2 - 1).y, mesh->get_node(h->nindex2 - 1).z, 0),
			make_float4(mesh->get_node(h->nindex3 - 1).x, mesh->get_node(h->nindex3 - 1).y, mesh->get_node(h->nindex3 - 1).z, 0),
			make_float4(mesh->get_node(h->nindex4 - 1).x, mesh->get_node(h->nindex4 - 1).y, mesh->get_node(h->nindex4 - 1).z, 0) }; // for every node xyz is required...shit


		GetExitTet(mesh->curr.o, mesh->curr.d, nodes, findex, adjtets, lastface, nextface, nexttet);

		fprintf(stderr, "Number of next tetrahedra: %lu \n", nexttet);
		fprintf(stderr, "Number of next face: %lu \n\n", nextface);

		if (mesh->get_face(nextface).face_is_constrained == true) { d.constrained = true; d.face = nextface; d.tet = nexttet; break; }
		if (mesh->get_face(nextface).face_is_wall == true) { d.wall = true; d.face = nextface; d.tet = nexttet; break; }
		if (nexttet == -1) { d.wall = true; d.face = nextface; d.tet = start; break; } // when adjacent tetrahedra is -1, ray stops
		lastface = nextface;
		start = nexttet;
	}
	if (d.wall == true) fprintf_s(stderr, "Wall hit.\n"); // finally... (27.10.2015)
	if (d.constrained == true) fprintf_s(stderr, "Triangle hit.\n"); // now i have: index of face(=triangle) which is intersected by ray.
}

int main()
{
	// load tetrahedral model
	tetrahedra_mesh tetmesh;
	tetmesh.load_tet_ele("untitled.1.ele");
	tetmesh.load_tet_neigh("untitled.1.neigh");
	tetmesh.load_tet_node("untitled.1.node");
	tetmesh.load_tet_face("untitled.1.face");
	tetmesh.load_tet_t2f("untitled.1.t2f");
	tetmesh.cam.d = make_float4(0, 1, 0, 0);
	tetmesh.cam.o = make_float4(0, 5, 5, 0);
	tetmesh.curr = tetmesh.cam;
	

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
	traverse_ray(&tetmesh, start, firsthit);

	system("PAUSE");
}


