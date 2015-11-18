#define NO_MSG //increases speed greatly
#include "tetgen_io.h"

const int width = 320, height=240;
const int spp = 4;


RGB trace(Ray r, tetrahedra_mesh *mesh, int32_t start, int depth)
{
	rayhit firsthit;
	traverse_ray(mesh, r, start, firsthit, depth);

	face fc = mesh->get_face(firsthit.face);
	node a1 = mesh->get_node(fc.node_a);
	node a2 = mesh->get_node(fc.node_b);
	node a3 = mesh->get_node(fc.node_c);
	double c = firsthit.dist;
	
	
	float k = ((255-0) / (0-10)); // in zweiter klammer erster wert ist untere grenze distanzwerte
	float d = 0 - (10 * k);

	float d2 = (c*k) + d;
	return RGB(c,0,d2); // return depth value
}




int main()
{
	// die ersten beiden von ray.d haben keine auswirkung - deshalb immer gleiche distanzwerte!!!!!!!!!!!!!!!!!!!!!!!

	// load tetrahedral model
	tetrahedra_mesh tetmesh;
	tetmesh.load_tet_ele("untitled.1.ele");
	tetmesh.load_tet_neigh("untitled.1.neigh");
	tetmesh.load_tet_node("untitled.1.node");
	tetmesh.load_tet_face("untitled.1.face");
	tetmesh.load_tet_t2f("untitled.1.t2f");
	tetmesh.cam.o = make_float4(1.8, 4, 3, 0);
	tetmesh.cam.d = make_float4(0.6, -1, 0, 0);
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
	FILE *f2 = fopen("test3.txt", "w");

	float4 camera_position = tetmesh.curr.o;
	float4 camera_direction = normalize(tetmesh.curr.d);
	float4 camera_up = make_float4(0, 0, 1, 0);
	float4 camera_right = Cross(camera_direction, camera_up);
	camera_up = Cross(camera_right, camera_direction); // This corrects for any slop in the choice of "up".


	RGB *color=new RGB[width*height];
	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			for (int s = 0; s < spp; s++){

				double normalized_x = (x / width) - 0.5;
				double normalized_y = (y / height) - 0.5;
				float4 image_point = camera_right * normalized_x + camera_up * normalized_y + camera_position + camera_direction;
				float4 ray_direction = image_point - camera_position;


				ray_direction.x = ray_direction.x + RND / 700;
				ray_direction.y = ray_direction.y + RND / 700;
				Ray rt;
				rt.d = ray_direction;
				rt.o = camera_position;

				RGB c = trace(rt, &tetmesh, start, 0);
				fprintf(f2, "%f %f %f \n", c.x,c.y,c.z);
				color[(height - y - 1)*width + x] = color[(height - y - 1)*width + x] + (c/spp);
			}
		}
	}

	// write to image
	FILE *f = fopen("image.ppm", "w");         // Write image to PPM file. 
	fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
	for (int i = 0; i<width*height; i++)
		fprintf(f, "%d %d %d ", (int)color[i].x, (int)color[i].y, (int)color[i].z);
}


