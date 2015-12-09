#define GLEW_STATIC
#include "GLFW/glfw3.h"
#include "tetgen_io.h"
#include "cuPrintf.cuh"
#include "device_launch_parameters.h"
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\extras\CUPTI\include\GL\glew.h"
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\extras\CUPTI\include\GL\glut.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

const int width = 640, height=480, spp = 4;
float3* cr;
float3* accumulatebuffer;
int frames = 0;
__device__ float gamma = 2.2f;
GLuint vbo;
mesh2 *mesh;

float4 cam_o = make_float4(-16, 5, -5, 0);
float4 cam_d = make_float4(0, 0, 0, 0);
float4 cam_u = make_float4(0, 0, 1, 0);

union Color  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

__device__ float timer = 0.0f;

unsigned int WangHash(unsigned int a) {
	// richiesams.blogspot.co.nz/2015/03/creating-randomness-and-acummulating.html
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

// CUDA error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		system("PAUSE");
		if (abort) exit(code);
	}
}


__device__ float getDepth(Ray r, mesh2 *mesh, rayhit firsthit)
{
	float4 a1 = make_float4(mesh->n_x[mesh->f_node_a[firsthit.face]], mesh->n_y[mesh->f_node_a[firsthit.face]], mesh->n_z[mesh->f_node_a[firsthit.face]], 0);
	float4 a2 = make_float4(mesh->n_x[mesh->f_node_b[firsthit.face]], mesh->n_y[mesh->f_node_b[firsthit.face]], mesh->n_z[mesh->f_node_b[firsthit.face]], 0);
	float4 a3 = make_float4(mesh->n_x[mesh->f_node_c[firsthit.face]], mesh->n_y[mesh->f_node_c[firsthit.face]], mesh->n_z[mesh->f_node_c[firsthit.face]], 0);
	float c = abs(intersect_dist(r, a1, a2, a3));
	float new_value = ((c - 0.f) / (80.f - 0.f)) * (1.f - 0.f) + 0.f;
	return new_value;
}


__device__ RGB radiance(Ray r, mesh2 *mesh, int32_t start, int depth)
{
	rayhit firsthit;
	traverse_ray(mesh, r, start, firsthit, depth);
	float d2 = getDepth(r, mesh, firsthit); // gets depth value
	RGB rd;
	rd.x = 0; rd.y = 0; rd.z = d2;
	return rd; 
}


__global__ void renderKernel(mesh2 *tetmesh, int32_t start, float4 cam_o, float4 cam_d, float4 cam_u, float3 *c, unsigned int hashedframenumber)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1)*width + x;

	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	curandState randState;
	curand_init(hashedframenumber + threadId, 0, 0, &randState);

	RGB c0(0);
	for (int s = 0; s < spp; s++)
	{
		float yu = 1.0f - ((y + curand_uniform(&randState)) / float(height - 1));
		float xu = (x + curand_uniform(&randState)) / float(width - 1);
		Ray ray = makeCameraRay(45.0f, cam_o, cam_d, cam_u, xu, yu);
		RGB rd = radiance(ray, tetmesh, start, 0);
		c0 = c0 + rd;
	}
	c0 = c0 / 4;

	Color fcolour;
	float3 colour = make_float3(clamp(c0.x, 0.0f, 1.0f), clamp(c0.y, 0.0f, 1.0f), clamp(c0.z, 0.0f, 1.0f));

	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / gamma) * 255), (unsigned char)(powf(colour.y, 1 / gamma) * 255), (unsigned char)(powf(colour.z, 1 / gamma) * 255), 1);
	c[i] = make_float3(x, y, fcolour.c);
}


void disp(void)
{
	frames++;
	cudaThreadSynchronize();
	cudaGLMapBufferObject((void**)&cr, vbo);
	glClear(GL_COLOR_BUFFER_BIT);

	dim3 block(8, 8, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	renderKernel << <grid, block >> >(mesh, start, cam_o, cam_d, cam_u, cr, WangHash(frames));
	gpuErrchk(cudaDeviceSynchronize());

	cudaGLUnmapBufferObject(vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, width * height);
	glDisableClientState(GL_VERTEX_ARRAY);
	glutSwapBuffers();
}



float mstep = 0.5f;
void keyFunc(unsigned char key, int x, int y)
{
	switch (key) {
	case 27: 
		// Escape
		printf("Done.\n");
		exit(0);
		break;
	case 'a': 
	{
				cam_o.y = cam_o.y + mstep;
				  break;
	}
	case 'd': 
	{
				cam_o.y = cam_o.y - mstep;
				  break;
	}
	case 'w': 
	{
				cam_o.x = cam_o.x + mstep;
				  break;
	}
	case 's': 
	{
				cam_o.x = cam_o.x - mstep;
				  break;
	}
	}
}

void idleFunc(void) {
	//UpdateRendering();

	glutPostRedisplay();
}

	
void render()
{
	char *argv[] = { "null", NULL };
	int   argc = 1;
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(width, height);
	glutCreateWindow("tetra_mesh");

	fprintf(stderr, "OpenGL successfully initialized \n");
	glutKeyboardFunc(keyFunc);
	glutIdleFunc(idleFunc);
	glutDisplayFunc(disp);
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		exit(0);
	}
	fprintf(stderr, "GLEW successfully initialized  \n");

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, width, 0.0, height);
	glutPostRedisplay();

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	unsigned int size = width * height * sizeof(float3);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGLRegisterBufferObject(vbo);
	fprintf(stderr, "VBO created  \n");
	fprintf(stderr, "Entering glutMainLoop...  \n");
	glutMainLoop();
}

int main(int argc, char *argv[])
{
	cudaDeviceProp  prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice(&dev, &prop);

	tetrahedra_mesh tetmesh;
	tetmesh.load_tet_ele("test1.1.ele");
	tetmesh.load_tet_neigh("test1.1.neigh");
	tetmesh.load_tet_node("test1.1.node");
	tetmesh.load_tet_face("test1.1.face");
	tetmesh.load_tet_t2f("test1.1.t2f");


	// ===========================
	//     mesh2
	// ===========================

	gpuErrchk(cudaMallocManaged(&mesh, sizeof(mesh2)));

	// INDICES
	mesh->edgenum = tetmesh.edgenum;
	mesh->facenum = tetmesh.facenum;
	mesh->nodenum = tetmesh.nodenum;
	mesh-> tetnum = tetmesh.tetnum;

	// NODES
	cudaMallocManaged(&mesh->n_index, mesh->nodenum*sizeof(uint32_t));
	for (auto i : tetmesh.nodes) mesh->n_index[i.index] = i.index;
	cudaMallocManaged(&mesh->n_x, mesh->nodenum*sizeof(float));
	cudaMallocManaged(&mesh->n_y, mesh->nodenum*sizeof(float));
	cudaMallocManaged(&mesh->n_z, mesh->nodenum*sizeof(float));
	for (auto i : tetmesh.nodes) mesh->n_x[i.index] = i.x;
	for (auto i : tetmesh.nodes) mesh->n_y[i.index] = i.y;
	for (auto i : tetmesh.nodes) mesh->n_z[i.index] = i.z;

	// FACES
	cudaMallocManaged(&mesh->f_index, mesh->facenum*sizeof(uint32_t));
	for (auto i : tetmesh.faces) mesh->f_index[i.index] = i.index;
	cudaMallocManaged(&mesh->f_node_a, mesh->facenum*sizeof(uint32_t));
	cudaMallocManaged(&mesh->f_node_b, mesh->facenum*sizeof(uint32_t));
	cudaMallocManaged(&mesh->f_node_c, mesh->facenum*sizeof(uint32_t));
	for (auto i : tetmesh.faces) mesh->f_node_a[i.index] = i.node_a;
	for (auto i : tetmesh.faces) mesh->f_node_b[i.index] = i.node_b;
	for (auto i : tetmesh.faces) mesh->f_node_c[i.index] = i.node_c;
	cudaMallocManaged(&mesh->face_is_constrained, mesh->facenum*sizeof(bool));
	cudaMallocManaged(&mesh->face_is_wall, mesh->facenum*sizeof(bool));
	for (auto i : tetmesh.faces) mesh->face_is_constrained[i.index] = i.face_is_constrained;
	for (auto i : tetmesh.faces) mesh->face_is_wall[i.index] = i.face_is_wall;

	// TETRAHEDRA
	cudaMallocManaged(&mesh->t_index, mesh->tetnum*sizeof(uint32_t));
	for (auto i : tetmesh.tetrahedras) mesh->t_index[i.number] = i.number;
	cudaMallocManaged(&mesh->t_findex1, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_findex2, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_findex3, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_findex4, mesh->tetnum*sizeof(int32_t));
	for (auto i : tetmesh.tetrahedras) mesh->t_findex1[i.number] = i.findex1;
	for (auto i : tetmesh.tetrahedras) mesh->t_findex2[i.number] = i.findex2;
	for (auto i : tetmesh.tetrahedras) mesh->t_findex3[i.number] = i.findex3;
	for (auto i : tetmesh.tetrahedras) mesh->t_findex4[i.number] = i.findex4;
	cudaMallocManaged(&mesh->t_nindex1, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_nindex2, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_nindex3, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_nindex4, mesh->tetnum*sizeof(int32_t));
	for (auto i : tetmesh.tetrahedras) mesh->t_nindex1[i.number] = i.nindex1;
	for (auto i : tetmesh.tetrahedras) mesh->t_nindex2[i.number] = i.nindex2;
	for (auto i : tetmesh.tetrahedras) mesh->t_nindex3[i.number] = i.nindex3;
	for (auto i : tetmesh.tetrahedras) mesh->t_nindex4[i.number] = i.nindex4;
	cudaMallocManaged(&mesh->t_adjtet1, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_adjtet2, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_adjtet3, mesh->tetnum*sizeof(int32_t));
	cudaMallocManaged(&mesh->t_adjtet4, mesh->tetnum*sizeof(int32_t));
	for (auto i : tetmesh.tetrahedras) mesh->t_adjtet1[i.number] = i.adjtet1;
	for (auto i : tetmesh.tetrahedras) mesh->t_adjtet2[i.number] = i.adjtet2;
	for (auto i : tetmesh.tetrahedras) mesh->t_adjtet3[i.number] = i.adjtet3;
	for (auto i : tetmesh.tetrahedras) mesh->t_adjtet4[i.number] = i.adjtet4;

	// ===========================
	//     mesh end
	// ===========================

	// Get bounding box
	BBox box = init_BBox(mesh);
	fprintf_s(stderr, "\nBounding box:MIN xyz - %f %f %f \n", box.min.x, box.min.y, box.min.z);
	fprintf_s(stderr, "             MAX xyz - %f %f %f \n\n", box.max.x, box.max.y, box.max.z);

	// Get camera starting tetrahedra
	gpuErrchk(cudaMallocManaged(&cr, width * height * sizeof(float3)));
	GetTetrahedraFromPoint << <mesh->tetnum, 1>> >(mesh, cam_o);
	gpuErrchk(cudaDeviceSynchronize()); 

	if (start == 0) 
	{
		fprintf(stderr, "Starting point outside tetrahedra! Aborting ... \n");
		system("PAUSE");
		exit(0);

	} else fprintf(stderr, "Starting tetrahedra - camera: %lu \n", start);
	
	// main render loop

	clock_t t1 = clock();
	render();
	clock_t t2 = clock();

	double t = (double)(t2 - t1) / CLOCKS_PER_SEC;
	printf("\nRender time: %fs.\n", t);

	cudaFree(mesh);
	cudaFree(cr);

	system("PAUSE");
}


