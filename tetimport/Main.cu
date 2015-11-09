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
	double d = intersect_dist(r, a1.f_node(), a2.f_node(), a3.f_node());

	int d2 = (((d - 0)*(255 - 0)) / (10 - 0)) + 0;
	return RGB(0,0,d2); // return depth value
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
	tetmesh.cam.o = make_float4(2.13, 5.08, 3.97, 0);
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


	// raytracing stuff

	RGB *color=new RGB[width*height];
	for (int x = 0; x < width; x++){
		for (int y = 0; y < height; y++){
			for (int s = 0; s < 4; s++){

				float4 cam = camcr(width, height, x, y);
				cam.x = cam.x + RND / 700;
				cam.y = cam.y + RND / 700;

				tetmesh.curr.d = normalize(cam - tetmesh.curr.o);

				color[(height - y - 1)*width + x] = color[(height - y - 1)*width + x] + (trace(tetmesh.curr, &tetmesh, start, 0)/4);
			}
		}
	}

	// write to image
	FILE *f = fopen("image.ppm", "w");         // Write image to PPM file. 
	fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
	for (int i = 0; i<width*height; i++)
		fprintf(f, "%d %d %d ", (int)color[i].x, (int)color[i].y, (int)color[i].z);
}


