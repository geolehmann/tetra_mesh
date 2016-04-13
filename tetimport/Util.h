#pragma once

#define STB_TRUETYPE_IMPLEMENTATION
#include <stdio.h>
#include <string>
#include "stb_truetype.h"
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\extras\CUPTI\include\GL\glew.h"

unsigned char ttf_buffer[1<<20];
unsigned char temp_bitmap[512*512];

stbtt_bakedchar cdata[96]; // ASCII 32..126 is 95 glyphs
GLuint ftex;

void my_stbtt_initfont(void)
{
   fread(ttf_buffer, 1, 1<<20, fopen("times.ttf", "rb"));
   stbtt_BakeFontBitmap(ttf_buffer,0, 32.0, temp_bitmap,512,512, 32,96, cdata); // no guarantee this fits!
   // can free ttf_buffer at this point
   glGenTextures(1, &ftex);
   glBindTexture(GL_TEXTURE_2D, ftex);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, 512,512, 0, GL_ALPHA, GL_UNSIGNED_BYTE, temp_bitmap);
   // can free temp_bitmap at this point
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

void my_stbtt_print(float x, float y, std::string str, float3 rgb)
{
   const char *text = str.c_str();
   glEnable(GL_BLEND);
   // assume orthographic projection with units = screen pixels, origin at top left
   glEnable(GL_TEXTURE_2D);
   glBindTexture(GL_TEXTURE_2D, ftex);
   glBegin(GL_QUADS);

   while (*text) {
      if (*text >= 32 && *text < 128) {
         stbtt_aligned_quad q;
         stbtt_GetBakedQuad(cdata, 512,512, *text-32, &x,&y,&q,1);//1=opengl & d3d10+,0=d3d9
		 glTexCoord2f(q.s0,q.t1); glVertex2f(q.x0, y - q.y1 + y); //a
         glTexCoord2f(q.s1,q.t1); glVertex2f(q.x1, y - q.y1 + y); //b
         glTexCoord2f(q.s1,q.t0); glVertex2f(q.x1, y - q.y0 + y); //c
         glTexCoord2f(q.s0,q.t0); glVertex2f(q.x0, y - q.y0 + y); //d
      }
      ++text;
   }
	 glEnd();
    glColor3f(rgb.x, rgb.y, rgb.z);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_BLEND);
}

