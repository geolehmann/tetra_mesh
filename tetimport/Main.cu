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
#include "tetgen_io.h"
#include "Camera.h"
#include "cuPrintf.cuh"
#include "device_launch_parameters.h"
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\extras\CUPTI\include\GL\glew.h"
#include "GLFW/glfw3.h"
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#define spp 1
#define gamma 1.2f
#define MAX_DEPTH 3
#define width 1024	
#define height 768

float3* finalimage;
float3* accumulatebuffer;
int32_t frameNumber = 0;
bool bufferReset = false;
float deltaTime, lastFrame;
BBox box;
GLuint vbo;
mesh2 *mesh;

// Camera
InteractiveCamera* interactiveCamera = NULL;
Camera* hostRendercam = NULL;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
bool buttonActive = false;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -30.0;
int lastX = width / 2, lastY = height / 2;
int theButtonState = 0;
int theModifierState = 0;
float scalefactor = 1.2f;

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
	// GLFW error callback
	fputs(description, stderr);
}

void updateCamPos()
{
	// check if current pos is still inside tetrahedralization
	CheckOutOfBBox(&box, hostRendercam->position);
	// look for new tetrahedra...
	uint32_t _dim = 2 + pow(mesh->tetnum, 0.25);
	dim3 Block(_dim, _dim, 1);
	dim3 Grid(_dim, _dim, 1);
	GetTetrahedraFromPoint << <Grid, Block >> >(mesh, hostRendercam->position);
	gpuErrchk(cudaDeviceSynchronize());
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	int dist = 1.0f;

	if (action == GLFW_PRESS) buttonActive = true;
	if (action == GLFW_RELEASE) buttonActive = false;

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
	if (key == GLFW_KEY_A && buttonActive)
	{
		interactiveCamera->strafe(-dist); updateCamPos();
	}
	if (key == GLFW_KEY_D && buttonActive)
	{
		interactiveCamera->strafe(dist); updateCamPos();
	}
	if (key == GLFW_KEY_W && buttonActive)
	{
		interactiveCamera->goForward(dist); updateCamPos();
	}
	if (key == GLFW_KEY_S && buttonActive)
	{
		interactiveCamera->goForward(-dist); updateCamPos();
	}

	if (key == GLFW_KEY_R && buttonActive)
	{
		interactiveCamera->changeAltitude(dist); updateCamPos();
	}
	if (key == GLFW_KEY_F && buttonActive)
	{
		interactiveCamera->changeAltitude(-dist); updateCamPos();
	}
	if (key == GLFW_KEY_G && buttonActive)
	{
		interactiveCamera->changeApertureDiameter(0.1);
	}
	if (key == GLFW_KEY_H && buttonActive)
	{
		interactiveCamera->changeApertureDiameter(-0.1);
	}
	if (key == GLFW_KEY_T && buttonActive)
	{
		interactiveCamera->changeFocalDistance(0.1);
	}
	if (key == GLFW_KEY_Z && buttonActive)
	{
		interactiveCamera->changeFocalDistance(-0.1);
	}

	if (key == GLFW_KEY_UP && buttonActive)
	{
		interactiveCamera->changePitch(0.02f);
	}
	if (key == GLFW_KEY_DOWN && buttonActive)
	{
		interactiveCamera->changePitch(-0.02f);
	}
	if (key == GLFW_KEY_LEFT && buttonActive)
	{
		interactiveCamera->changeYaw(0.02f);
	}
	if (key == GLFW_KEY_RIGHT && buttonActive)
	{
		interactiveCamera->changeYaw(-0.02f);
	}

	bufferReset = true;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) theButtonState = 0;
	if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) theButtonState = 1;
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) theButtonState = 2;
}


void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{	
	int deltaX = lastX - xpos;
	int deltaY = lastY - ypos;

	if (deltaX != 0 || deltaY != 0) {

		if (theButtonState == 0)  // Rotate
		{
			interactiveCamera->changeYaw(deltaX * 0.01);
			interactiveCamera->changePitch(-deltaY * 0.01);
		}
		else if (theButtonState == 1) // Zoom
		{
			interactiveCamera->changeAltitude(-deltaY * 0.01);
			updateCamPos();
		}

		if (theButtonState == 2) // camera move
		{
			interactiveCamera->changeRadius(-deltaY * 0.01);
			updateCamPos();
		}

		lastX = xpos;
		lastY = ypos;
		bufferReset = true;
	}
}

__device__ RGB radiance(mesh2 *mesh, int32_t start, Ray &ray, float4 oldpos, curandState* randState)
{
	float4 mask = make_float4(1.0f, 1.0f, 1.0f, 0.0f);	// colour mask
	float4 accucolor = make_float4(0.0f, 0.0f, 0.0f, 0.0f);	// accumulated colour
	float4 originInWorldSpace = ray.o;
	float4 rayInWorldSpace = ray.d;
	int32_t newstart = start;

	for (int depth = 0; depth < MAX_DEPTH; depth++)
	{
		float4 f = make_float4(0, 0, 0, 0);  // primitive colour
		float4 emit = make_float4(0, 0, 0, 0); // primitive emission colour
		float4 x; // intersection point
		float4 n; // normal
		float4 nl; // oriented normal
		float4 dw; // ray direction of next path segment
		float4 pointHitInWorldSpace;
		float3 rayorig = make_float3(originInWorldSpace.x, originInWorldSpace.y, originInWorldSpace.z);
		float3 raydir = make_float3(rayInWorldSpace.x, rayInWorldSpace.y, rayInWorldSpace.z);

		rayhit firsthit;
		traverse_ray(mesh, originInWorldSpace, rayInWorldSpace, newstart, firsthit);
		// set new starting tetrahedra and ray origin
		float4 a1 = make_float4(mesh->n_x[mesh->f_node_a[firsthit.face]], mesh->n_y[mesh->f_node_a[firsthit.face]], mesh->n_z[mesh->f_node_a[firsthit.face]], 0);
		float4 a2 = make_float4(mesh->n_x[mesh->f_node_b[firsthit.face]], mesh->n_y[mesh->f_node_b[firsthit.face]], mesh->n_z[mesh->f_node_b[firsthit.face]], 0);
		float4 a3 = make_float4(mesh->n_x[mesh->f_node_c[firsthit.face]], mesh->n_y[mesh->f_node_c[firsthit.face]], mesh->n_z[mesh->f_node_c[firsthit.face]], 0);
		// get intersection distance
		float t = intersect_dist(ray, a1, a2, a3);

		x = originInWorldSpace + rayInWorldSpace * t;
		n = normalize(getTriangleNormal(a1, a2, a3));
		nl = Dot(n, rayInWorldSpace) < 0 ? n : n * -1;  // correctly oriented normal

		if (firsthit.constrained == true) { emit = make_float4(6.0f, 4.0f, 1.0f, 0.0f); f = make_float4(0.5f, 0.3f, 0.7f, 0.0f); }
		if (firsthit.wall == true) { emit = make_float4(0.1f, 0.1f, 0.1f, 0.0f); f = make_float4(0.6f, 0.5f, 0.4f, 0.0f); }
		if (firsthit.dark == true) { emit = make_float4(1.0f, 1.0f, 0.0f, 0.0f); f = make_float4(1.0f, 0.0f, 0.0f, 0.0f); }

		accucolor += (mask * emit);

		firsthit.refl_t = DIFF;

		// basic material system, all parameters are hard-coded (such as phong exponent, index of refraction)

		// diffuse material, based on smallpt by Kevin Beason 
		if (firsthit.refl_t == DIFF){

			// pick two random numbers
			float phi = 2 * PI * curand_uniform(randState);
			float r2 = curand_uniform(randState);
			float r2s = sqrtf(r2);

			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			float4 w = nl; w = normalize(w);
			float4 u = Cross((fabs(w.x) > .1 ? make_float4(0, 1, 0, 0) : make_float4(1, 0, 0, 0)), w); u = normalize(u);
			float4 v = Cross(w, u);

			// compute cosine weighted random ray direction on hemisphere 
			dw = u*cosf(phi)*r2s + v*sinf(phi)*r2s + w*sqrtf(1 - r2);
			dw = normalize(dw);

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x +w * 0.01;  // scene size dependent

			// multiply mask with colour of object
			mask *= f;
		}

		// Phong metal material from "Realistic Ray Tracing", P. Shirley
		if (firsthit.refl_t == METAL){

			// compute random perturbation of ideal reflection vector
			// the higher the phong exponent, the closer the perturbed vector is to the ideal reflection direction
			float phi = 2 * PI * curand_uniform(randState);
			float r2 = curand_uniform(randState);
			float phongexponent = 20;
			float cosTheta = powf(1 - r2, 1.0f / (phongexponent + 1));
			float sinTheta = sqrtf(1 - cosTheta * cosTheta);

			// create orthonormal basis uvw around reflection vector with hitpoint as origin 
			// w is ray direction for ideal reflection
			float4 w = rayInWorldSpace - n * 2.0f * Dot(n, rayInWorldSpace); w = normalize(w);
			float4 u = Cross((fabs(w.x) > .1 ? make_float4(0, 1, 0, 0) : make_float4(1, 0, 0, 0)), w); u = normalize(u);
			float4 v = Cross(w, u); // v is normalised by default

			// compute cosine weighted random ray direction on hemisphere 
			dw = u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta;
			dw = normalize(dw);

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + w * 0.01;  // scene size dependent

			// multiply mask with colour of object
			mask *= f;
		}

		// specular material (perfect mirror)
		if (firsthit.refl_t == SPEC){

			// compute reflected ray direction according to Snell's law
			dw = rayInWorldSpace - n * 2.0f * Dot(n, rayInWorldSpace);

			// offset origin next path segment to prevent self intersection
			pointHitInWorldSpace = x + nl * 0.01;   // scene size dependent

			// multiply mask with colour of object
			mask *= f;
		}

		// perfectly refractive material (glass, water)
		if (firsthit.refl_t == REFR){

			bool into = Dot(n, nl) > 0; // is ray entering or leaving refractive material?
			float nc = 1.0f;  // Index of Refraction air
			float nt = 1.5f;  // Index of Refraction glass/water
			float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
			float ddn = Dot(rayInWorldSpace, nl);
			float cos2t = 1.0f - nnt*nnt * (1.f - ddn*ddn);

			if (cos2t < 0.0f) // total internal reflection 
			{
				dw = rayInWorldSpace;
				dw -= n * 2.0f * Dot(n, rayInWorldSpace);

				// offset origin next path segment to prevent self intersection
				pointHitInWorldSpace = x + nl * 0.01; // scene size dependent
			}
			else // cos2t > 0
			{
				// compute direction of transmission ray
				float4 tdir = rayInWorldSpace * nnt;
				tdir -= n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t)));
				tdir = normalize(tdir);

				float R0 = (nt - nc)*(nt - nc) / (nt + nc)*(nt + nc);
				float c = 1.f - (into ? -ddn : Dot(tdir, n));
				float Re = R0 + (1.f - R0) * c * c * c * c * c;
				float Tr = 1 - Re; // Transmission
				float P = .25f + .5f * Re;
				float RP = Re / P;
				float TP = Tr / (1.f - P);

				// randomly choose reflection or transmission ray
				if (curand_uniform(randState) < 0.25) // reflection ray
				{
					mask *= RP;
					dw = rayInWorldSpace;
					dw -= n * 2.0f * Dot(n, rayInWorldSpace);

					pointHitInWorldSpace = x + nl * 0.01; // scene size dependent
				}
				else // transmission ray
				{
					mask *= TP;
					dw = tdir; //r = Ray(x, tdir); 
					pointHitInWorldSpace = x + nl * 0.001f; // epsilon must be small to avoid artefacts
				}
			}
		}

		// set up origin and direction of next path segment
		originInWorldSpace = pointHitInWorldSpace;
		rayInWorldSpace = dw;
		newstart = firsthit.tet; // new tet origin
	}
	// add radiance up to a certain ray depth
	// return accumulated ray colour after all bounces are computed
	return RGB(accucolor.x, accucolor.y, accucolor.z);
}


__global__ void renderKernel(mesh2 *tetmesh, int32_t start, float3 *accumbuffer, float3 *c, unsigned int hashedframenumber, unsigned int framenumber, float4 position, float4 view, float4 up, float fovx, float fovy, float focalDistance, float apertureRadius)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1)*width + x;

	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	curandState randState;
	curand_init(hashedframenumber + threadId, 0, 0, &randState);

	int pixelx = x; // pixel x-coordinate on screen
	int pixely = height - y - 1; // pixel y-coordintate on screen
	float4 rendercampos = make_float4(position.x, position.y, position.z, 0);
	RGB finalcol(0);

	for (int s = 0; s < spp; s++)
	{
		float4 rendercamview = make_float4(view.x, view.y, view.z, 0); rendercamview = normalize(rendercamview); // view is already supposed to be normalized, but normalize it explicitly just in case.
		float4 rendercamup = make_float4(up.x, up.y, up.z, 0); rendercamup = normalize(rendercamup);
		float4 horizontalAxis = Cross(rendercamview, rendercamup); horizontalAxis = normalize(horizontalAxis); // Important to normalize!
		float4 verticalAxis = Cross(horizontalAxis, rendercamview); verticalAxis = normalize(verticalAxis); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.
		float4 middle = rendercampos + rendercamview;
		float4 horizontal = horizontalAxis * tanf(fovx * 0.5 * (PI / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.
		float4 vertical = verticalAxis * tanf(-fovy * 0.5 * (PI / 180)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5. I also normzlized A and B, so there's no need to divide by the length of A or B anymore. Also normalized view and removed lengthOfView. Also removed the cast to float.
		float jitterValueX = curand_uniform(&randState) - 0.5;
		float jitterValueY = curand_uniform(&randState) - 0.5;
		float sx = (jitterValueX + pixelx) / (width - 1);
		float sy = (jitterValueY + pixely) / (height - 1);
		float4 pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
		float4 pointOnImagePlane = rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * focalDistance); // Important for depth of field!		
		float4 aperturePoint;
		if (apertureRadius > 0.00001)
		{ 
			float random1 = curand_uniform(&randState);
			float random2 = curand_uniform(&randState);
			float angle = 2 * PI * random1;
			float distance = apertureRadius * sqrtf(random2);
			float apertureX = cos(angle) * distance;
			float apertureY = sin(angle) * distance;
			aperturePoint = rendercampos + (horizontalAxis * apertureX) + (verticalAxis * apertureY);
		}
		else { 	aperturePoint = rendercampos;	}

		// calculate ray direction of next ray in path
		float4 apertureToImagePlane = pointOnImagePlane - aperturePoint;
		apertureToImagePlane = normalize(apertureToImagePlane); // ray direction, needs to be normalised
		float4 rayInWorldSpace = apertureToImagePlane;
		rayInWorldSpace = normalize(rayInWorldSpace);
		float4 originInWorldSpace = aperturePoint;

		finalcol += radiance(tetmesh, start, Ray(originInWorldSpace, rayInWorldSpace), rendercampos, &randState) * (1.0f/spp);
	}

	accumbuffer[i] += finalcol;
	float3 tempcol = accumbuffer[i] / framenumber;

	Color fcolour;
	float3 colour = make_float3(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f));

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
	glfwSetMouseButtonCallback(window, mouse_button_callback);
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
	glOrtho(0.0, width, 0.0, height, 0, 1);
	
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(float3), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGraphicsResource_t _cgr;
	size_t num_bytes;
	cudaGraphicsGLRegisterBuffer(&_cgr, vbo, cudaGraphicsRegisterFlagsNone);
	fprintf(stderr, "VBO created  \n");
	fprintf(stderr, "Entering glutMainLoop...  \n");

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		if (bufferReset)
		{
			frameNumber = 0;
			cudaMemset(accumulatebuffer, 1, width * height * sizeof(float3));
		}
		bufferReset = false;
		frameNumber++;
		interactiveCamera->buildRenderCamera(hostRendercam);

		// Calculate deltatime of current frame
		GLfloat currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		std::stringstream title;
		title << "tetra_mesh (2015)   -   deltaTime: " << deltaTime*1000 << " ms. (16-36 optimal)";
		glfwSetWindowTitle(window, title.str().c_str());
		
		// CUDA interop
		cudaGraphicsMapResources(1, &_cgr, 0);
		cudaGraphicsResourceGetMappedPointer((void**)&finalimage, &num_bytes,_cgr);
		glClear(GL_COLOR_BUFFER_BIT);
		dim3 block(16, 16, 1);
		dim3 grid(width / block.x, height / block.y, 1);
		renderKernel << <grid, block >> >(mesh, _start_tet, accumulatebuffer, finalimage, WangHash(frameNumber), frameNumber, 
		hostRendercam->position, hostRendercam->view, hostRendercam->up, hostRendercam->fov.x, hostRendercam->fov.x, 
		hostRendercam->focalDistance, hostRendercam->apertureRadius);
		gpuErrchk(cudaDeviceSynchronize());
		cudaGraphicsUnmapResources(1, &_cgr, 0);

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
	delete interactiveCamera;
	interactiveCamera = new InteractiveCamera();
	interactiveCamera->setResolution(width, height);
	interactiveCamera->setFOVX(45);
	hostRendercam = new Camera();
	interactiveCamera->buildRenderCamera(hostRendercam);

	cudaDeviceProp prop;
	int dev;
	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice(&dev, &prop);

	tetrahedra_mesh tetmesh;
	tetmesh.load_tet_ele("test5.1.ele");
	tetmesh.load_tet_neigh("test5.1.neigh");
	tetmesh.load_tet_node("test5.1.node");
	tetmesh.load_tet_face("test5.1.face");
	tetmesh.load_tet_t2f("test5.1.t2f");


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

	// Allocate unified memory
	gpuErrchk(cudaMallocManaged(&finalimage, width * height * sizeof(float3)));
	gpuErrchk(cudaMallocManaged(&accumulatebuffer, width * height * sizeof(float3)));

	// find starting tetrahedra
	uint32_t _dim = 2+pow(mesh->tetnum, 0.25);
	dim3 Block(_dim, _dim, 1);
	dim3 Grid(_dim, _dim, 1);
	GetTetrahedraFromPoint << <Grid, Block >> >(mesh, hostRendercam->position);
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


