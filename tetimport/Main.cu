/*
*  tetrahedra-based raytracer
*  Copyright (C) 2015  Christian Lehmann
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*/

#define GLEW_STATIC
#include <stdio.h>
#include "tetgen_io.h"
#include "cuPrintf.cuh"
#include "device_launch_parameters.h"
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\extras\CUPTI\include\GL\glew.h"
#include "GLFW/glfw3.h"
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

const int width = 640, height=480, spp = 4;
float3* cr;
int frames = 0;
__device__ float gamma = 2.2f;
__device__ float fov = 40.0f;
BBox box;
GLuint vbo;
mesh2 *mesh;

#define MAX_DEPTH 3

// Camera
bool keys[1024];
GLfloat sensitivity = 0.15f;
bool firstMouse = true;
float4 cam_o = make_float4(-14, 11, 11, 0);
float4 cam_d = make_float4(0, 0, 0, 0);
float4 cam_u = make_float4(0, 0, 1, 0);
GLfloat Yaw = 90.0f;	// horizontal inclination
GLfloat Pitch = 0.0f; // vertikal inclination
GLfloat lastX = width / 2.0; //screen center
GLfloat lastY = height / 2.0;
GLfloat deltaTime = 0.0f;	// Time between current frame and last frame
GLfloat lastFrame = 0.0f;

union Color  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};


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

unsigned int WangHash(unsigned int a) {
	// richiesams.blogspot.co.nz/2015/03/creating-randomness-and-acummulating.html
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

static void error_callback(int error, const char* description)
{
	fputs(description, stderr);
}


void updateCamPos()
{
	//look for new tetrahedra...
	uint32_t _dim = 2 + pow(mesh->tetnum, 0.25);
	dim3 Block(_dim, _dim, 1);
	dim3 Grid(_dim, _dim, 1);
	GetTetrahedraFromPoint << <Grid, Block >> >(mesh, cam_o);
	gpuErrchk(cudaDeviceSynchronize());
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	// first value sets velocity
	GLfloat cameraSpeed = 0.5f * deltaTime;
	if (key >= 0 && key < 1024)
	{
		if (action == GLFW_PRESS)
		{
			keys[key] = true;
		}
		else if (action == GLFW_RELEASE)
		{
			keys[key] = false;
		}

	}
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
		
	if (keys[GLFW_KEY_A])
	{
		updateCamPos;
		cam_o -= normalizeCPU(CrossCPU(minus(cam_d, cam_o), cam_u)) * cameraSpeed;
	}
	if (keys[GLFW_KEY_D])
	{
		updateCamPos;
		cam_o += normalizeCPU(CrossCPU(minus(cam_d, cam_o), cam_u)) * cameraSpeed;
	}
	if (keys[GLFW_KEY_W])
	{
		updateCamPos;
		cam_o += minus(cam_d, cam_o) * cameraSpeed;
	}
	if (keys[GLFW_KEY_S])
	{
		updateCamPos;
		cam_o -= minus(cam_d, cam_o) * cameraSpeed;
	}
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{	
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}
	GLfloat xoffset = xpos - lastX;
	GLfloat yoffset = lastY - ypos; // Reversed since y-coordinates go from bottom to left
	
	lastX = xpos;
	lastY = ypos;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	Yaw += yoffset; //geändert - vorher y/x vertauscht
	Pitch += xoffset;

	// Make sure that when pitch is out of bounds, screen doesn't get flipped
	if (Pitch > 89.0f)
		Pitch = 89.0f;
	if (Pitch < -89.0f)
		Pitch = -89.0f;

	float4 front;
	front.x = cos(radian(Yaw)) * cos(radian(Pitch));
	front.y = sin(radian(Pitch));
	front.z = sin(radian(Yaw)) * cos(radian(Pitch));
	cam_d = normalizeCPU(front);
}



__device__ float getDepth(Ray r, mesh2 *mesh, int32_t face)
{
	float4 a1 = make_float4(mesh->n_x[mesh->f_node_a[face]], mesh->n_y[mesh->f_node_a[face]], mesh->n_z[mesh->f_node_a[face]], 0);
	float4 a2 = make_float4(mesh->n_x[mesh->f_node_b[face]], mesh->n_y[mesh->f_node_b[face]], mesh->n_z[mesh->f_node_b[face]], 0);
	float4 a3 = make_float4(mesh->n_x[mesh->f_node_c[face]], mesh->n_y[mesh->f_node_c[face]], mesh->n_z[mesh->f_node_c[face]], 0);
	float c = abs(intersect_dist(r, a1, a2, a3));
	float new_value = ((c - 0.0f) / (100.0f - 0.0f)) * (1.0f - 0.0f) + 0.0f;
	return new_value;
}

__device__ float4 getTriangleNormal(const float4 &p1, const float4 &p2, const float4 &p3)
{
	float4 V1 = (p2 - p1);
	float4 V2 = (p3 - p1);
	float4 surfaceNormal;
	surfaceNormal.x = (V1.y*V2.z) - (V1.z - V2.y);
	surfaceNormal.y = -((V2.z * V1.x) - (V2.x * V1.z));
	surfaceNormal.z = (V1.x*V2.y) - (V1.y*V2.x);
	surfaceNormal = normalize(surfaceNormal);
	return surfaceNormal;
}

__device__ RGB visualizeDepth(Ray r, mesh2 *mesh, int32_t start, int depth)
{
	rayhit firsthit;
	traverse_ray(mesh, r, start, firsthit, depth);
	float d2 = getDepth(r, mesh, firsthit.face); // gets depth value

	RGB rd;
	if (firsthit.wall == true) { rd.x = 0.5; rd.y = 0.8; rd.z = 0.1; }
	if (firsthit.constrained == true) { rd.x = 0.1; rd.y = 0.1; rd.z = d2; }
	return rd; 
}


__device__ RGB radiance(mesh2 *mesh, int32_t &start, float4 ray_o, float4 ray_d)
{
	Ray r;
	float4 mask = make_float4(1.0f, 1.0f, 1.0f, 0.0f);	// colour mask
	float4 accucolor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);	// accumulated colour
	float4 f;  // primitive colour
	float4 emit; // primitive emission colour
	float4 x; // intersection point
	float4 n; // normal
	float4 nl; // oriented normal
	float4 d; // ray direction of next path segment
	r.d = ray_d;
	r.o = ray_o;

	for (int depth = 1; depth <= MAX_DEPTH; depth++)
	{

		rayhit firsthit;
		traverse_ray(mesh, r, start, firsthit, depth);
		// set new starting tetrahedra and ray origin
		float4 a1 = make_float4(mesh->n_x[mesh->f_node_a[firsthit.face]], mesh->n_y[mesh->f_node_a[firsthit.face]], mesh->n_z[mesh->f_node_a[firsthit.face]], 0);
		float4 a2 = make_float4(mesh->n_x[mesh->f_node_b[firsthit.face]], mesh->n_y[mesh->f_node_b[firsthit.face]], mesh->n_z[mesh->f_node_b[firsthit.face]], 0);
		float4 a3 = make_float4(mesh->n_x[mesh->f_node_c[firsthit.face]], mesh->n_y[mesh->f_node_c[firsthit.face]], mesh->n_z[mesh->f_node_c[firsthit.face]], 0);
		float t = abs(intersect_dist(r, a1, a2, a3));

		x = r.o + r.d*t;  // intersection point
		n = normalize(getTriangleNormal(a1, a2, a3));  // normal 
		nl = Dot(n, r.d) < 0 ? n : n * -1;  // correctly oriented normal
		f = make_float4(0.9f, 0.4f, 0.1f, 0.0f);  // triangle colour
		emit = make_float4(0.6f, 0.6f, 0.6f, 0.0f);
		accucolor += (mask * emit);


		firsthit.refl = SPEC;


		if (firsthit.refl == SPEC)
		{
			// compute relfected ray direction according to Snell's law
			d = r.d - 2.0f * n * Dot(n, r.d);
			// offset origin next path segment to prevent self intersection
			x += nl * 0.01f;
			// multiply mask with colour of object
			mask *= f;
		}
		r.o = x;
		r.d = d;
		start = firsthit.tet; // new tet origin
	}
	RGB rgb;
	rgb.x = accucolor.x;
	rgb.y = accucolor.y;
	rgb.z = accucolor.z;
	return rgb;
}


__global__ void renderKernel(mesh2 *tetmesh, int32_t start, float4 cam_o, float4 cam_d, float4 cam_u, float3 *c, unsigned int hashedframenumber)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1)*width + x;

	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	curandState randState;
	curand_init(hashedframenumber + threadId, 0, 0, &randState);

	RGB c0(0,0,0);
	for (int s = 0; s < spp; s++)
	{
		float yu = 1.0f - ((y + curand_uniform(&randState)) / float(height - 1));
		float xu = (x + curand_uniform(&randState)) / float(width - 1);
		Ray ray = makeCameraRay(fov, cam_o, cam_d, cam_u, xu, yu);
		RGB rd = visualizeDepth(ray, tetmesh, start, 0);
		// RGB rd = radiance(tetmesh, start, ray.o, ray.d);
		c0 = c0 + rd;
	}
	c0 = c0 / 4;

	Color fcolour;
	float3 colour = make_float3(clamp(c0.x, 0.0f, 1.0f), clamp(c0.y, 0.0f, 1.0f), clamp(c0.z, 0.0f, 1.0f));

	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / gamma) * 255), (unsigned char)(powf(colour.y, 1 / gamma) * 255), (unsigned char)(powf(colour.z, 1 / gamma) * 255), 1);
	c[i] = make_float3(x, y, fcolour.c);
}


void render()
{
	GLFWwindow* window;
	if (!glfwInit()) exit(EXIT_FAILURE);
	window = glfwCreateWindow(width, height, "", NULL, NULL);
	glfwMakeContextCurrent(window);
	glfwSetErrorCallback(error_callback);
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glewExperimental = GL_TRUE;
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) 
	{
		fprintf(stderr, "GLEW not supported.");
		fflush(stderr);
		exit(0);
	}
	fprintf(stderr, "GLEW successfully initialized  \n");


	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, width, 0.0, height, 0, 1);
	
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(float3), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGLRegisterBufferObject(vbo);
	fprintf(stderr, "VBO created  \n");
	fprintf(stderr, "Entering glutMainLoop...  \n");

	while (!glfwWindowShouldClose(window))
	{
		// Calculate deltatime of current frame
		GLfloat currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		std::stringstream title;
		title << "tetra_mesh (2015)   -   deltaTime: " << deltaTime*1000 << " ms. (16-36 optimal)";
		glfwSetWindowTitle(window, title.str().c_str());

		glClear(GL_COLOR_BUFFER_BIT);
		glfwPollEvents();
		cudaGLMapBufferObject((void**)&cr, vbo);

		dim3 block(8, 8, 1);
		dim3 grid(width / block.x, height / block.y, 1);
		renderKernel << <grid, block >> >(mesh, _start_tet, cam_o, cam_d, cam_u, cr, WangHash(frames));
		gpuErrchk(cudaDeviceSynchronize());

		cudaGLUnmapBufferObject(vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexPointer(2, GL_FLOAT, 12, 0);
		glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);
		glDrawArrays(GL_POINTS, 0, width * height);
		glDisableClientState(GL_VERTEX_ARRAY);

		glfwSwapBuffers(window);
	}
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
	box = init_BBox(mesh);
	fprintf_s(stderr, "\nBounding box:MIN xyz - %f %f %f \n", box.min.x, box.min.y, box.min.z);
	fprintf_s(stderr, "             MAX xyz - %f %f %f \n\n", box.max.x, box.max.y, box.max.z);

	// Get camera starting tetrahedra
	gpuErrchk(cudaMallocManaged(&cr, width * height * sizeof(float3)));

	// grid dimensions for finding starting tetrahedra
	uint32_t _dim = 2+pow(mesh->tetnum, 0.25);
	dim3 Block(_dim, _dim, 1);
	dim3 Grid(_dim, _dim, 1);
	GetTetrahedraFromPoint << <Grid, Block >> >(mesh, cam_o);
	gpuErrchk(cudaDeviceSynchronize()); 

	if (_start_tet == 0) 
	{
		fprintf(stderr, "Starting point outside tetrahedra! Aborting ... \n");
		system("PAUSE");
		exit(0);

	} else fprintf(stderr, "Starting tetrahedra - camera: %lu \n", _start_tet);
	
	// main render function

	render();

	gpuErrchk(cudaDeviceReset());
	glfwTerminate();
	return 0;
}


