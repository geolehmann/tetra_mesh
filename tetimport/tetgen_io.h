#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <deque>
#include <array>

#include "Math.h"

struct node // nodes have an index and three coordinates
{
	uint32_t index;
	float x, y, z;
};

struct edge // an edge consists of two node indices and the index of the edge
{
	uint32_t index;
	uint32_t node1, node2;
};

struct face //each face has three node indices, an own index and information whether it is constrained(=model triangle) or wall
{
	uint32_t index;
	uint32_t node_a, node_b, node_c;
	bool face_is_constrained=false;
	bool face_is_wall=false;
};


class tetrahedra // tetrahedras have indexes for the tetrahedra itself and for faces, nodes and neighbor(adjecent) tetrahedra
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
	BBox boundingbox;
	Ray cam;

	void load_tet_neigh(std::string filename);
	void load_tet_ele(std::string filename);
	void load_tet_node(std::string filename);
	void load_tet_face(std::string filename);
	void load_tet_t2f(std::string filename);
	void load_tet_edge(std::string filename);

	void load_tetmodel(std::string filename);


	bool IsPointInTetrahedron(tetrahedra t, float4 p);
	int32_t GetTetrahedraFromPoint(float4 p);
	void init_BBox();
	tetrahedra get_tetrahedra(uint32_t t){ return tetrahedras.at(t - 1); }
	face get_face(uint32_t t){ return faces.at(t); }

private:
	std::deque<tetrahedra>tetrahedras;
	std::deque<node>nodes;
	std::deque<face>faces;
	std::deque<edge>edges;
	uint32_t max = 1000000000;

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
				tetrahedras.at(ints.at(0) - 1).number = ints.at(0); //nummer von aktuellem tetrahedra
				tetrahedras.at(ints.at(0) - 1).nindex1 = ints.at(1);
				tetrahedras.at(ints.at(0) - 1).nindex2 = ints.at(2);
				tetrahedras.at(ints.at(0) - 1).nindex3 = ints.at(3);
				tetrahedras.at(ints.at(0) - 1).nindex4 = ints.at(4);
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
				tetrahedras.at(ints.at(0) - 1).adjtet1 = ints.at(1);
				tetrahedras.at(ints.at(0) - 1).adjtet2 = ints.at(2);
				tetrahedras.at(ints.at(0) - 1).adjtet3 = ints.at(3);
				tetrahedras.at(ints.at(0) - 1).adjtet4 = ints.at(4);
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
				nodes.at((int)ints.at(0) - 1).index = ints.at(0);
				nodes.at((int)ints.at(0) - 1).x = ints.at(1);
				nodes.at((int)ints.at(0) - 1).y = ints.at(2);
				nodes.at((int)ints.at(0) - 1).z = ints.at(3);
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
				faces.at(ints.at(0) - 1).index = ints.at(0);
				faces.at(ints.at(0) - 1).node_a = ints.at(1);
				faces.at(ints.at(0) - 1).node_a = ints.at(2);
				faces.at(ints.at(0) - 1).node_a = ints.at(3);

				if (ints.at(5) == -1 || ints.at(6) == -1) { faces.at(ints.at(0) - 1).face_is_wall = true; }
				
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
				edges.resize(edgenum, ed1); //Tetrahedra-Deque f�llen
			}
			else if (ints.size() != NULL) // restliche Zeilen
			{
				edges.at(ints.at(0) - 1).index = ints.at(0);
				edges.at(ints.at(0) - 1).node1 = ints.at(1);
				edges.at(ints.at(0) - 1).node1 = ints.at(2);
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
			if (num == 0) //Erste Zeile
			{
				//nada
			}
			else if (ints.size() != NULL) // restliche Zeilen
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


bool tetrahedra_mesh::IsPointInTetrahedron(tetrahedra t, float4 p)
{
	float4 v1 = make_float4(nodes.at(t.nindex1 - 1).x, nodes.at(t.nindex1 - 1).y, nodes.at(t.nindex1 - 1).z, 0);
	float4 v2 = make_float4(nodes.at(t.nindex2 - 1).x, nodes.at(t.nindex2 - 1).y, nodes.at(t.nindex2 - 1).z, 0);
	float4 v3 = make_float4(nodes.at(t.nindex3 - 1).x, nodes.at(t.nindex3 - 1).y, nodes.at(t.nindex3 - 1).z, 0);
	float4 v4 = make_float4(nodes.at(t.nindex4 - 1).x, nodes.at(t.nindex4 - 1).y, nodes.at(t.nindex4 - 1).z, 0);

	return SameSide(v1, v2, v3, v4, p) &&
		SameSide(v2, v3, v4, v1, p) &&
		SameSide(v3, v4, v1, v2, p) &&
		SameSide(v4, v1, v2, v3, p);
}


void GetExitTet(float4 ray_o, float4 ray_d, float4 nodes[4], int32_t findex[4], int32_t adjtet[4], int32_t lface, int32_t &face, int32_t &tet)
{
	// http://realtimecollisiondetection.net/blog/?p=13

	// translate Ray to origin and vertices same as ray
	float4 q = ray_d - ray_o;

	float4 v1 = make_float4(nodes[1].x,nodes[1].y,nodes[1].z, 0); // A
	float4 v2 = make_float4(nodes[2].x, nodes[2].y, nodes[2].z, 0); // B
	float4 v3 = make_float4(nodes[3].x, nodes[3].y, nodes[3].z, 0); // C
	float4 v4 = make_float4(nodes[4].x, nodes[4].y, nodes[4].z, 0); // D

	float4 p[4];
	p[0] = v1 - ray_o;
	p[1] = v2 - ray_o;
	p[2] = v3 - ray_o;
	p[3] = v4 - ray_o;

	// ABC
	if (signf(ScTP(q, p[0], p[1])) == signf(ScTP(q, p[1], p[2])) && signf(ScTP(q, p[1], p[2])) == signf(ScTP(q, p[2], p[0])) && lface != findex[4]) { face = findex[4]; tet = adjtet[4]; }
	// BAD
	if (signf(ScTP(q, p[1], p[0])) == signf(ScTP(q, p[0], p[3])) && signf(ScTP(q, p[0], p[3])) == signf(ScTP(q, p[3], p[1])) && lface != findex[3]) { face = findex[3]; tet = adjtet[3]; }
	// CDA
	if (signf(ScTP(q, p[2], p[3])) == signf(ScTP(q, p[3], p[0])) && signf(ScTP(q, p[3], p[0])) == signf(ScTP(q, p[0], p[2])) && lface != findex[2]) { face = findex[2]; tet = adjtet[2]; }
	// DCB
	if (signf(ScTP(q, p[3], p[2])) == signf(ScTP(q, p[2], p[1])) && signf(ScTP(q, p[2], p[1])) == signf(ScTP(q, p[1], p[3])) && lface != findex[1]) { face = findex[1]; tet = adjtet[1]; }
	printf("Error! \n"); 
	face = 0;
	tet = 0;
}



int32_t tetrahedra_mesh::GetTetrahedraFromPoint(float4 p)
{
	for (auto t : tetrahedras)
	{
		if (IsPointInTetrahedron(t, p) == true) return t.number;
	}
	return -1;
}

void tetrahedra_mesh::init_BBox()
{
	boundingbox.min = make_float4(-1000000000, -1000000000, -1000000000, 0);
	boundingbox.max = make_float4(1000000000, 1000000000, 1000000000, 0);
	for (auto n : nodes)
	{
		if (boundingbox.min.x < n.x)  boundingbox.min.x = n.x;
		if (boundingbox.max.x > n.x)  boundingbox.max.x = n.x;
		if (boundingbox.min.y < n.y)  boundingbox.min.y = n.y;
		if (boundingbox.max.y > n.y)  boundingbox.max.y = n.y;
		if (boundingbox.min.z < n.z)  boundingbox.min.z = n.z;
		if (boundingbox.max.z > n.z)  boundingbox.max.z = n.z;
	}
}

