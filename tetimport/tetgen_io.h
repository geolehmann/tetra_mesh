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

#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <ctime>

#include "Math.h"

__managed__ int32_t _start_tet = 0;

#define inf 1e20

struct node
{
	uint32_t index;
	float x, y, z;
	float4 f_node(){ return make_float4(x, y, z, 0); }
};

struct edge
{
	uint32_t index;
	uint32_t node1, node2;
};

struct face
{
	uint32_t index;
	uint32_t node_a, node_b, node_c;
	bool face_is_constrained = false;
	bool face_is_wall = false;
};


class tetrahedra
{
public:
	uint32_t number;
	int32_t findex1, findex2, findex3, findex4;
	int32_t nindex1, nindex2, nindex3, nindex4;
	int32_t adjtet1, adjtet2, adjtet3, adjtet4;
};

class tetrahedra_mesh
{
public:
	uint32_t tetnum, nodenum, facenum, edgenum;
	std::deque<tetrahedra>tetrahedras;
	std::deque<node>nodes;
	std::deque<face>faces;
	std::deque<edge>edges;
	uint32_t max = 1000000000;

	void load_tet_neigh(std::string filename);
	void load_tet_ele(std::string filename);
	void load_tet_node(std::string filename);
	void load_tet_face(std::string filename);
	void load_tet_t2f(std::string filename);
	void load_tet_edge(std::string filename);
};

void tetrahedra_mesh::load_tet_ele(std::string filename)
{
	uint32_t num = 0;
	tetrahedra tet1;
	std::string line;
	std::ifstream myfile(filename);
	if (myfile.is_open())
	{
		while (std::getline(myfile, line) && num<max) // Nur die ersten tausend Zeilen einlesen
		{
			std::stringstream in(line);
			std::vector<int32_t> ints;
			copy(std::istream_iterator<int32_t, char>(in), std::istream_iterator<int32_t, char>(), back_inserter(ints));
			if (num == 0) //Erste Zeile
			{
				tetnum = ints.at(0); //In erster Zeile der .ele-Datei ist Anzahl der Tetraheder abgelegt
				tetrahedras.resize(tetnum, tet1); //Tetrahedra-Deque füllen
			}
			else if (ints.size() != NULL) // restliche Zeilen
			{
				tetrahedras.at(ints.at(0)).number = ints.at(0); //nummer von aktuellem tetrahedra
				tetrahedras.at(ints.at(0)).nindex1 = ints.at(1);
				tetrahedras.at(ints.at(0)).nindex2 = ints.at(2);
				tetrahedras.at(ints.at(0)).nindex3 = ints.at(3);
				tetrahedras.at(ints.at(0)).nindex4 = ints.at(4);
			}
			num++;
		}
		myfile.close();
	}
	else std::cout << "Unable to open .ele file";
	fprintf_s(stderr, "Total number of tetrahedra in .ele-file: %u \n", num);
}


void tetrahedra_mesh::load_tet_neigh(std::string filename)
{
	uint32_t num = 0;
	std::string line;
	std::ifstream myfile(filename);
	if (myfile.is_open())
	{
		while (std::getline(myfile, line) && num<max) // Nur die ersten tausend Zeilen einlesen
		{
			std::stringstream in(line);
			std::vector<int32_t> ints;
			copy(std::istream_iterator<int32_t, char>(in), std::istream_iterator<int32_t, char>(), back_inserter(ints));
			if (num != 0 && ints.size() != NULL)
			{
				tetrahedras.at(ints.at(0)).adjtet1 = ints.at(1);
				tetrahedras.at(ints.at(0)).adjtet2 = ints.at(2);
				tetrahedras.at(ints.at(0)).adjtet3 = ints.at(3);
				tetrahedras.at(ints.at(0)).adjtet4 = ints.at(4);
			}
			num++;
		}
		myfile.close();
	}
	else std::cout << "Unable to open .neigh file";
	fprintf_s(stderr, "Total number of tetrahedra in .neigh-file: %u \n", num);
}



void tetrahedra_mesh::load_tet_node(std::string filename)
{
	uint32_t num = 0;
	node nd1;
	std::string line;
	std::ifstream myfile(filename);
	if (myfile.is_open())
	{
		while (std::getline(myfile, line) && num<max) // Nur die ersten tausend Zeilen einlesen
		{
			std::stringstream in(line);
			std::vector<float> ints;
			copy(std::istream_iterator<float, char>(in), std::istream_iterator<float, char>(), back_inserter(ints));
			if (num == 0) //Erste Zeile
			{
				nodenum = int(ints.at(0)); //In erster Zeile der .ele-Datei ist Anzahl der Tetraheder abgelegt
				nodes.resize(nodenum, nd1); //Tetrahedra-Deque füllen
			}
			else if (ints.size() != NULL) // restliche Zeilen
			{
				nodes.at((int)ints.at(0)).index = ints.at(0);
				nodes.at((int)ints.at(0)).x = ints.at(1);
				nodes.at((int)ints.at(0)).y = ints.at(2);
				nodes.at((int)ints.at(0)).z = ints.at(3);
			}
			num++;
		}
		myfile.close();
	}
	else std::cout << "Unable to open .node file";
	fprintf_s(stderr, "Total number of Nodes in .node-file: %u \n", num);
}


void tetrahedra_mesh::load_tet_face(std::string filename)
{
	uint32_t num = 0;
	face fc1;
	std::string line;
	std::ifstream myfile(filename);
	if (myfile.is_open())
	{
		while (std::getline(myfile, line) && num<max) // Nur die ersten tausend Zeilen einlesen
		{
			std::stringstream in(line);
			std::vector<int32_t> ints;
			copy(std::istream_iterator<int32_t, char>(in), std::istream_iterator<int32_t, char>(), back_inserter(ints));
			if (num == 0) //Erste Zeile
			{
				facenum = int(ints.at(0)); //In erster Zeile der .ele-Datei ist Anzahl der Tetraheder abgelegt
				faces.resize(facenum, fc1); //Tetrahedra-Deque füllen
			}
			else if (ints.size() != NULL) // restliche Zeilen
			{
				faces.at(ints.at(0)).index = ints.at(0);
				faces.at(ints.at(0)).node_a = ints.at(1);
				faces.at(ints.at(0)).node_b = ints.at(2);
				faces.at(ints.at(0)).node_c = ints.at(3);

				

				if (ints.at(5) == -1 || ints.at(6) == -1) { faces.at(ints.at(0)).face_is_wall = true; }
				else if (ints.at(4) == -1) faces.at(ints.at(0)).face_is_constrained = true;
			}
			num++;
		}
		myfile.close();
	}
	else std::cout << "Unable to open .face file";
	fprintf_s(stderr, "Total number of Faces in .face-file: %u \n", num);
}



void tetrahedra_mesh::load_tet_edge(std::string filename)
{
	uint32_t num = 0;
	edge ed1;
	std::string line;
	std::ifstream myfile(filename);
	if (myfile.is_open())
	{
		while (std::getline(myfile, line) && num<max) // Nur die ersten tausend Zeilen einlesen
		{
			std::stringstream in(line);
			std::vector<int32_t> ints;
			copy(std::istream_iterator<int32_t, char>(in), std::istream_iterator<int32_t, char>(), back_inserter(ints));
			if (num == 0) //Erste Zeile
			{
				edgenum = int(ints.at(0)); //In erster Zeile der .ele-Datei ist Anzahl der Tetraheder abgelegt
				edges.resize(edgenum, ed1); //Tetrahedra-Deque füllen
			}
			else if (ints.size() != NULL) // restliche Zeilen
			{
				edges.at(ints.at(0)).index = ints.at(0);
				edges.at(ints.at(0)).node1 = ints.at(1);
				edges.at(ints.at(0)).node1 = ints.at(2);
			}
			num++;
		}
		myfile.close();
	}
	else std::cout << "Unable to open .edge file";
	fprintf_s(stderr, "Total number of Edges in .edge-file: %u \n", num);
}


void tetrahedra_mesh::load_tet_t2f(std::string filename)
{
	uint32_t num = 0;
	std::string line;
	std::ifstream myfile(filename);
	if (myfile.is_open())
	{
		while (std::getline(myfile, line) && num<max) // Nur die ersten tausend Zeilen einlesen
		{
			std::stringstream in(line);
			std::vector<int32_t> ints;
			copy(std::istream_iterator<int32_t, char>(in), std::istream_iterator<int32_t, char>(), back_inserter(ints));

			if (ints.size() != NULL) // alle Zeilen
			{
				tetrahedras.at(ints.at(0) - 1).findex1 = ints.at(1);
				tetrahedras.at(ints.at(0) - 1).findex2 = ints.at(2);
				tetrahedras.at(ints.at(0) - 1).findex3 = ints.at(3);
				tetrahedras.at(ints.at(0) - 1).findex4 = ints.at(4);
			}
			num++;
		}
		myfile.close();
	}
	else std::cout << "Unable to open .t2f file";
	fprintf_s(stderr, "Total number of Tetrahedra in .t2f-file: %u \n", num);
}

//--------------------------------------------------------------------------------------------------------------------------------------




__device__ bool IsPointInTetrahedron(float4 v1, float4 v2, float4 v3, float4 v4, float4 p)
{
	return SameSide(v1, v2, v3, v4, p) &&
		SameSide(v2, v3, v4, v1, p) &&
		SameSide(v3, v4, v1, v2, p) &&
		SameSide(v4, v1, v2, v3, p);
}

__global__ void GetTetrahedraFromPoint(mesh2* mesh, float4 p)
{
		int i = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

		float4 v1 = make_float4(mesh->n_x[mesh->t_nindex1[i]], mesh->n_y[mesh->t_nindex1[i]], mesh->n_z[mesh->t_nindex1[i]], 0);
		float4 v2 = make_float4(mesh->n_x[mesh->t_nindex2[i]], mesh->n_y[mesh->t_nindex2[i]], mesh->n_z[mesh->t_nindex2[i]], 0);
		float4 v3 = make_float4(mesh->n_x[mesh->t_nindex3[i]], mesh->n_y[mesh->t_nindex3[i]], mesh->n_z[mesh->t_nindex3[i]], 0);
		float4 v4 = make_float4(mesh->n_x[mesh->t_nindex4[i]], mesh->n_y[mesh->t_nindex4[i]], mesh->n_z[mesh->t_nindex4[i]], 0);
		if (i<mesh->tetnum) if (IsPointInTetrahedron(v1, v2, v3, v4, p) == true) _start_tet=i;

}

BBox init_BBox(mesh2* mesh)
{
	BBox boundingbox;
	boundingbox.min = make_float4(-inf, -inf, -inf, 0);
	boundingbox.max = make_float4(inf, inf, inf, 0);
	for (uint32_t i = 0; i < mesh->nodenum; i++)
	{
		if (boundingbox.min.x < mesh->n_x[i])  boundingbox.min.x = mesh->n_x[i];
		if (boundingbox.max.x > mesh->n_x[i])  boundingbox.max.x = mesh->n_x[i];
		if (boundingbox.min.y < mesh->n_y[i])  boundingbox.min.y = mesh->n_y[i];
		if (boundingbox.max.y > mesh->n_y[i])  boundingbox.max.y = mesh->n_y[i];
		if (boundingbox.min.z < mesh->n_z[i])  boundingbox.min.z = mesh->n_z[i];
		if (boundingbox.max.z > mesh->n_z[i])  boundingbox.max.z = mesh->n_z[i];
	}
	return boundingbox;
}

void CheckOutOfBBox(BBox* boundingbox, float4 &p)
{
	if (boundingbox->min.x + 0.2 < p.x)  p.x = boundingbox->min.x;
	if (boundingbox->max.x - 0.2 > p.x)  p.x = boundingbox->max.x;
	if (boundingbox->min.y + 0.2 < p.y)  p.y = boundingbox->min.y;
	if (boundingbox->max.y - 0.2 > p.y)  p.y = boundingbox->max.y;
	if (boundingbox->min.z + 0.2 < p.z)  p.z = boundingbox->min.z;
	if (boundingbox->max.z - 0.2 > p.z)  p.z = boundingbox->max.z;
}


__device__ void GetExitTet(float4 ray_o, float4 ray_d, float4* nodes, int32_t findex[4], int32_t adjtet[4], int32_t lface, int32_t &face, int32_t &tet)
{
	face = 0;
	tet = 0;

	// http://realtimecollisiondetection.net/blog/?p=13

	// translate Ray to origin and vertices same as ray
	ray_d = ray_o + (ray_d * 1000);

	float4 q = ray_d - ray_o;

	float4 v1 = make_float4(nodes[0].x, nodes[0].y, nodes[0].z, 0); // A
	float4 v2 = make_float4(nodes[1].x, nodes[1].y, nodes[1].z, 0); // B
	float4 v3 = make_float4(nodes[2].x, nodes[2].y, nodes[2].z, 0); // C
	float4 v4 = make_float4(nodes[3].x, nodes[3].y, nodes[3].z, 0); // D

	float4 p[4] = { v1 - ray_o, v2 - ray_o, v3 - ray_o, v4 - ray_o };

	float u_3 = ScTP(q, p[0], p[1]);
	float v_3 = ScTP(q, p[1], p[2]);
	float w_3 = ScTP(q, p[2], p[0]);

	float u_2 = ScTP(q, p[1], p[0]);
	float v_2 = ScTP(q, p[0], p[3]);
	float w_2 = ScTP(q, p[3], p[1]);

	float u_1 = ScTP(q, p[2], p[3]);
	float v_1 = ScTP(q, p[3], p[0]);
	float w_1 = ScTP(q, p[0], p[2]);

	float u_0 = ScTP(q, p[3], p[2]);
	float v_0 = ScTP(q, p[2], p[1]);
	float w_0 = ScTP(q, p[1], p[3]);

	// ScTP funktioniert auch mit float4.
	// ABC
	if (lface != findex[3]) { if (signf(u_3) == signf(v_3) && signf(v_3) == signf(w_3)) { face = findex[3]; tet = adjtet[3]; } }
	// BAD
	if (lface != findex[2]) { if (signf(u_2) == signf(v_2) && signf(v_2) == signf(w_2)) { face = findex[2]; tet = adjtet[2]; } }
	// CDA
	if (lface != findex[1]) { if (signf(u_1) == signf(v_1) && signf(v_1) == signf(w_1)) { face = findex[1]; tet = adjtet[1]; } }
	// DCB
	if (lface != findex[0]) { if (signf(u_0) == signf(v_0) && signf(v_0) == signf(w_0)) { face = findex[0]; tet = adjtet[0]; } }
	// No face hit
	// if (face == 0 && tet == 0) { printf("Error! No exit tet found. \n"); }
}

__device__ void traverse_ray(mesh2 *mesh, Ray ray, int32_t start, rayhit &d, int depth)
{
	int32_t idx = start;
	int32_t nexttet, nextface, lastface = 0;
	while (1)
	{
		int32_t findex[4] = { mesh->t_findex1[idx], mesh->t_findex2[idx], mesh->t_findex3[idx], mesh->t_findex4[idx] };
		int32_t adjtets[4] = { mesh->t_adjtet1[idx], mesh->t_adjtet2[idx], mesh->t_adjtet3[idx], mesh->t_adjtet4[idx] };
		float4 nodes[4] = {
			make_float4(mesh->n_x[mesh->t_nindex1[idx]], mesh->n_y[mesh->t_nindex1[idx]], mesh->n_z[mesh->t_nindex1[idx]], 0),
			make_float4(mesh->n_x[mesh->t_nindex2[idx]], mesh->n_y[mesh->t_nindex2[idx]], mesh->n_z[mesh->t_nindex2[idx]], 0),
			make_float4(mesh->n_x[mesh->t_nindex3[idx]], mesh->n_y[mesh->t_nindex3[idx]], mesh->n_z[mesh->t_nindex3[idx]], 0),
			make_float4(mesh->n_x[mesh->t_nindex4[idx]], mesh->n_y[mesh->t_nindex4[idx]], mesh->n_z[mesh->t_nindex4[idx]], 0) };


		GetExitTet(ray.o, ray.d, nodes, findex, adjtets, lastface, nextface, nexttet);

		/*if (nexttet == 0 || nextface == 0)
		{
			d.wall = true;
			d.face = lastface;
			break;
		}*/
		depth++;

		if (mesh->face_is_constrained[nextface] == true) { d.constrained = true; d.face = nextface; d.tet = idx; break; }
		if (mesh->face_is_wall[nextface] == true) { d.wall = true; d.face = nextface; d.tet = idx; break; }
		if (nexttet == -1 || nextface == -1) { d.wall = true; d.face = nextface; d.tet = idx; break; } // when adjacent tetrahedra is -1, ray stops
		lastface = nextface;
		idx = nexttet;
		if (depth > 80) 
		{
			//avoid infinite loops
			d.wall = true;
			d.face = lastface;
			d.tet = idx;
			break;
		}
	}
}


