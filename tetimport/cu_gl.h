#include "GL/glut.h"
#include "GL/glext.h"
#include <cuda_runtime.h>
#include "cuda_gl_interop.h"
#include <iostream>

#define DIM 1024
#define GET_PROC_ADDRESS( str ) wglGetProcAddress( str )
PFNGLBINDBUFFERARBPROC    glBindBuffer = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData = NULL;

struct GPUAnimBitmap
{
	GLuint  bufferObj;
	cudaGraphicsResource *resource;
	int     width, height;
	void    *dataBlock;
	void(*fAnim)(uchar4*, void*, int);
	void(*animExit)(void*);
	void(*clickDrag)(void*, int, int, int, int);
	int     dragStartX, dragStartY;

	GPUAnimBitmap(int w, int h, void *d = NULL)
	{
		width = w;
		height = h;
		dataBlock = d;
		clickDrag = NULL;

		cudaDeviceProp  prop;
		int dev;
		memset(&prop, 0, sizeof(cudaDeviceProp));
		prop.major = 1;
		prop.minor = 0;
		cudaChooseDevice(&dev, &prop);
		cudaGLSetGLDevice(dev);

		int c = 1;
		char* dummy = "";
		glutInit(&c, &dummy);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
		glutInitWindowSize(width, height);
		glutCreateWindow("bitmap");

		glBindBuffer = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
		glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
		glGenBuffers = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
		glBufferData = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

		glGenBuffers(1, &bufferObj);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
		glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4,
			NULL, GL_DYNAMIC_DRAW_ARB);

		cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
	}

	~GPUAnimBitmap()
	{
		free_resources();
	}

	void free_resources(void)
	{
		cudaGraphicsUnregisterResource(resource);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		glDeleteBuffers(1, &bufferObj);
	}


	long image_size(void) const { return width * height * 4; }

	void click_drag(void(*f)(void*, int, int, int, int))
	{
		clickDrag = f;
	}

	void anim_and_exit(void(*f)(uchar4*, void*, int), void(*e)(void*))
	{
		GPUAnimBitmap**   bitmap = get_bitmap_ptr();
		*bitmap = this;
		fAnim = f;
		animExit = e;

		glutKeyboardFunc(Key);
		glutDisplayFunc(Draw);
		if (clickDrag != NULL)
			glutMouseFunc(mouse_func);
		glutIdleFunc(idle_func);
		glutMainLoop();
	}

	static GPUAnimBitmap** get_bitmap_ptr(void)
	{
		static GPUAnimBitmap*   gBitmap;
		return &gBitmap;
	}

	static void mouse_func(int button, int state, int mx, int my)
	{
		if (button == GLUT_LEFT_BUTTON) {
			GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
			if (state == GLUT_DOWN) {
				bitmap->dragStartX = mx;
				bitmap->dragStartY = my;
			}
			else if (state == GLUT_UP) {
				bitmap->clickDrag(bitmap->dataBlock,
					bitmap->dragStartX,
					bitmap->dragStartY,
					mx, my);
			}
		}
	}

	static void idle_func(void)
	{
		static int ticks = 1;
		GPUAnimBitmap*  bitmap = *(get_bitmap_ptr());
		uchar4*         devPtr;
		size_t  size;

		cudaGraphicsMapResources(1, &(bitmap->resource), NULL);
		cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, bitmap->resource);

		bitmap->fAnim(devPtr, bitmap->dataBlock, ticks++);

		cudaGraphicsUnmapResources(1, &(bitmap->resource), NULL);

		glutPostRedisplay();
	}

	static void Key(unsigned char key, int x, int y)
	{
		switch (key) {
		case 27:
			GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
			if (bitmap->animExit)
				bitmap->animExit(bitmap->dataBlock);
			bitmap->free_resources();
			exit(0);
		}
	}

	static void Draw(void)
	{
		GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(bitmap->width, bitmap->height, GL_RGBA,
			GL_UNSIGNED_BYTE, 0);
		glutSwapBuffers();
	}
};



__global__ void kernel(uchar4 *ptr, int ticks)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// now calculate the value at that position
	float fx = x - DIM / 2;
	float fy = y - DIM / 2;
	float d = sqrtf(fx * fx + fy * fy);
	unsigned char grey = (unsigned char)(128.0f + 127.0f *
		cos(d / 10.0f - ticks / 7.0f) /
		(d / 10.0f + 1.0f));
	ptr[offset].x = grey;
	ptr[offset].y = grey;
	ptr[offset].z = grey;
	ptr[offset].w = 255;
}

void generate_frame(uchar4 *pixels, void*, int ticks)
{
	dim3    grids(DIM / 16, DIM / 16);
	dim3    threads(16, 16);
	kernel << <grids, threads >> >(pixels, ticks);
}

int render()
{
	GPUAnimBitmap  bitmap(DIM, DIM, NULL);

	bitmap.anim_and_exit(generate_frame, NULL);
	return 0;
}
