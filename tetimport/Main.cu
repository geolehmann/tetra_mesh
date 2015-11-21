#include "tetgen_io.h"

const int width = 320, height=240, spp = 4;

float getDepth(Ray r, mesh2 *mesh, rayhit firsthit)
{
	float4 a1 = make_float4(mesh->n_x[mesh->f_node_a[firsthit.face]], mesh->n_y[mesh->f_node_a[firsthit.face]], mesh->n_z[mesh->f_node_a[firsthit.face]], 0);
	float4 a2 = make_float4(mesh->n_x[mesh->f_node_b[firsthit.face]], mesh->n_y[mesh->f_node_b[firsthit.face]], mesh->n_z[mesh->f_node_b[firsthit.face]], 0);
	float4 a3 = make_float4(mesh->n_x[mesh->f_node_c[firsthit.face]], mesh->n_y[mesh->f_node_c[firsthit.face]], mesh->n_z[mesh->f_node_c[firsthit.face]], 0);
	float c = abs(intersect_dist(r, a1, a2, a3));
	float k = ((255 - 0) / (0 - 80)); // in zweiter klammer erster wert ist untere grenze distanzwerte
	float d = 0 - (80 * k);
	return (c*k) + d;
}


RGB radiance(Ray r, mesh2 *mesh, int32_t start, int depth)
{
	rayhit firsthit;
	traverse_ray(mesh, r, start, firsthit, depth);
	float d2 = getDepth(r, mesh, firsthit); // return depth value
	return RGB(0,0,d2); 
}

void renderKernel(mesh2 *tetmesh, int32_t start, float4 cam_o, float4 cam_d,float4 cam_u)
{
	// raytracing stuff
	RGB c;
	FILE *f = fopen("image.ppm", "w");
	fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);


#pragma omp parallel for schedule(dynamic)
	for (int y = 0; y < height; ++y){
		for (int x = 0; x < width; ++x){
			for (int s = 0; s < spp; s++)
			{
				float yu = 1.0f - ((y + RND2) / float(height - 1));
				float xu = (x + RND2) / float(width - 1);
				Ray ray = makeCameraRay(45.0f, cam_o, cam_d, cam_u, xu, yu);
				c = c + radiance(ray, tetmesh, start, 0);
			}
			c = c / 4;
			fprintf(f, "%d %d %d ", (int)c.x, (int)c.y, (int)c.z);
		}
	}


}


int main()
{
	tetrahedra_mesh tetmesh;
	tetmesh.load_tet_ele("test1.1.ele");
	tetmesh.load_tet_neigh("test1.1.neigh");
	tetmesh.load_tet_node("test1.1.node");
	tetmesh.load_tet_face("test1.1.face");
	tetmesh.load_tet_t2f("test1.1.t2f");

	float4 cam_o = make_float4(-16, 5, -5, 0);
	float4 cam_d = make_float4(0, 0, 0, 0);
	float4 cam_u = make_float4(0, 0, 1, 0);

	
	// ===========================
	//     mesh2 testing
	// ===========================

	mesh2 *mesh2;
	cudaMallocManaged((void**)&mesh2, sizeof(mesh2));

	// ELEMENT INDICES
	mesh2->edgenum = tetmesh.edgenum;
	mesh2->facenum = tetmesh.facenum;
	mesh2->nodenum = tetmesh.nodenum;
	mesh2-> tetnum = tetmesh.tetnum;

	// NODES
	mesh2->n_index = new uint32_t[mesh2->nodenum];
	for (auto i : tetmesh.nodes) mesh2->n_index[i.index] = i.index;
	mesh2->n_x = new float[mesh2->nodenum];
	mesh2->n_y = new float[mesh2->nodenum];
	mesh2->n_z = new float[mesh2->nodenum];
	for (auto i : tetmesh.nodes) mesh2->n_x[i.index] = i.x;
	for (auto i : tetmesh.nodes) mesh2->n_y[i.index] = i.y;
	for (auto i : tetmesh.nodes) mesh2->n_z[i.index] = i.z;

	// FACES
	mesh2->f_index = new uint32_t[mesh2->facenum];
	for (auto i : tetmesh.faces) mesh2->f_index[i.index] = i.index;
	mesh2->f_node_a = new uint32_t[mesh2->facenum];
	mesh2->f_node_b = new uint32_t[mesh2->facenum];
	mesh2->f_node_c = new uint32_t[mesh2->facenum];
	for (auto i : tetmesh.faces) mesh2->f_node_a[i.index] = i.node_a;
	for (auto i : tetmesh.faces) mesh2->f_node_b[i.index] = i.node_b;
	for (auto i : tetmesh.faces) mesh2->f_node_c[i.index] = i.node_c;
	mesh2->face_is_constrained = new bool[mesh2->facenum];
	mesh2->face_is_wall = new bool[mesh2->facenum];
	for (auto i : tetmesh.faces) mesh2->face_is_constrained[i.index] = i.face_is_constrained;
	for (auto i : tetmesh.faces) mesh2->face_is_wall[i.index] = i.face_is_wall;

	// TETRAHEDRA
	mesh2->t_index = new uint32_t[mesh2->tetnum];
	for (auto i : tetmesh.tetrahedras) mesh2->t_index[i.number] = i.number;
	mesh2->t_findex1 = new int32_t[mesh2->tetnum];
	mesh2->t_findex2 = new int32_t[mesh2->tetnum];
	mesh2->t_findex3 = new int32_t[mesh2->tetnum];
	mesh2->t_findex4 = new int32_t[mesh2->tetnum];
	for (auto i : tetmesh.tetrahedras) mesh2->t_findex1[i.number] = i.findex1;
	for (auto i : tetmesh.tetrahedras) mesh2->t_findex2[i.number] = i.findex2;
	for (auto i : tetmesh.tetrahedras) mesh2->t_findex3[i.number] = i.findex3;
	for (auto i : tetmesh.tetrahedras) mesh2->t_findex4[i.number] = i.findex4;
	mesh2->t_nindex1 = new int32_t[mesh2->tetnum];
	mesh2->t_nindex2 = new int32_t[mesh2->tetnum];
	mesh2->t_nindex3 = new int32_t[mesh2->tetnum];
	mesh2->t_nindex4 = new int32_t[mesh2->tetnum];
	for (auto i : tetmesh.tetrahedras) mesh2->t_nindex1[i.number] = i.nindex1;
	for (auto i : tetmesh.tetrahedras) mesh2->t_nindex2[i.number] = i.nindex2;
	for (auto i : tetmesh.tetrahedras) mesh2->t_nindex3[i.number] = i.nindex3;
	for (auto i : tetmesh.tetrahedras) mesh2->t_nindex4[i.number] = i.nindex4;
	mesh2->t_adjtet1 = new int32_t[mesh2->tetnum];
	mesh2->t_adjtet2 = new int32_t[mesh2->tetnum];
	mesh2->t_adjtet3 = new int32_t[mesh2->tetnum];
	mesh2->t_adjtet4 = new int32_t[mesh2->tetnum];
	for (auto i : tetmesh.tetrahedras) mesh2->t_adjtet1[i.number] = i.adjtet1;
	for (auto i : tetmesh.tetrahedras) mesh2->t_adjtet2[i.number] = i.adjtet2;
	for (auto i : tetmesh.tetrahedras) mesh2->t_adjtet3[i.number] = i.adjtet3;
	for (auto i : tetmesh.tetrahedras) mesh2->t_adjtet4[i.number] = i.adjtet4;

	// ===========================
	//     mesh2 end
	// ===========================


	// Get bounding box
	BBox box = init_BBox(mesh2);
	fprintf_s(stderr, "\nBounding box:MIN xyz - %f %f %f \n", box.min.x, box.min.y, box.min.z);
	fprintf_s(stderr, "             MAX xyz - %f %f %f \n\n", box.max.x, box.max.y, box.max.z);

	// Find camera tetrahedra
	int32_t start;
	start = GetTetrahedraFromPoint(mesh2, cam_o);
	if (start == -1) {
		fprintf_s(stderr, "Camera point not inside mesh! Aborting\n");
		system("PAUSE");
		return 0;
	}
	else 
	{ 
		fprintf_s(stderr, "Starting point (camera) tetra number: %lu\n\n", start); 
		fprintf_s(stderr, "Raytracing started... \n\n");
	}


	clock_t t1 = clock();
	renderKernel(mesh2, start,cam_o,cam_d,cam_u);
	clock_t t2 = clock();
	double t = (double)(t2 - t1) / CLOCKS_PER_SEC;
	printf("\nRender time: %fs.\n", t);
	system("PAUSE");;
}


