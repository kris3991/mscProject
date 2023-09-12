#include "../include/componentMananger.cuh"

class Edge
{
public:
	int v0;
	int v1;
	Edge() : v0(-1), v1(-1) {};
	Edge(int a, int b) : v0(a), v1(b) {};
};

struct halfEdgeStruct
{
public:
	int edge;
	int vertex;
	int halfEdge;
	int next;
	int face;
	bool boundary;
	
};

class HalfEdge
{
	//takes in the data read from fileManagement class.

protected:
	HalfEdge();

	static HalfEdge* he;

public:

   
    std::vector<int> halfEdges;
    std::vector<int> triangleAdj;
	//for debugging
	std::map<int, std::vector<int>> edgeFaceMap;
	std::map<int, std::vector<int>> faceVertexMap;
	std::map<int, Edge> edgeMap;
	//end
	std::map<int, int> heMap;
	std::map<int, halfEdgeStruct> halfEdgeDs;
	//vertex-half edge hash map
	std::map<int, std::vector<halfEdgeStruct>> vertexHalfEdgeMap;
	//edge-half edge hash map
	std::map<int, halfEdgeStruct> edgeHalfEdgeMap;
	//face-half edge hash map.
	std::map<int, std::vector<halfEdgeStruct>> faceHalfEdgeMap;
	std::string query;
	//the following face is for reading the values from the obj file. has vertices, normals and textures.

    
	HalfEdge(HalfEdge& other) = delete;
	void operator=(const HalfEdge&) = delete; 
	static HalfEdge* GetInstance();
	void clear();
    void initialise(TriangleMesh* tm);
	//the information gets written into the file.
	void calculateVertexNormals(std::string fileName);
	void adjascencyInformation(std::string fileName, TriangleMesh* tm);
	void calculateOneRing(int vertex, TriangleMesh* tm);
	
};