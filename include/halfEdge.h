#include "../include/componentMananger.cuh"

class HalfEdgeStruct;

class HalfEdgeStruct {
public:
	int vertex;
	int face;
	int next;
	int previous;
	// If twin == vertex, it's a border edge.
	int twin;

	// Constructor
	HalfEdgeStruct() : vertex(-1), face(-1), next(0), previous(0), twin(-1) {}
};



class HalfEdge
{
	//takes in the data read from fileManagement class.

protected:
	HalfEdge();

	static HalfEdge* he;


public:
	//for a pair for vertices, set an index.
	std::map<std::pair<int, int>, int> edgeStructure;
	std::map<int, std::pair<int, int>> edgeStructure2;
	//we use hash map for o(1) access.
	std::map<int, HalfEdgeStruct> hES;
	std::map<int, std::vector<int>> vertexHes;
	std::map<int, bool> normalsCalculated;
	//adjascency matrix for bfs.
	std::map<int, std::set<int>> adjascencyList;
	//queue for bfs
	std::queue<int> bfsQueue;
	//visited map.
	std::map<int, bool> visited;
	//distance
	std::map<int, int> geodesicDistance;
	//processing time for geodesic
	std::chrono::duration<double> geodesicProcessingTime;
	//calculation time.
	std::chrono::duration<double> geodesicCalcTime;
    
	HalfEdge(HalfEdge& other) = delete;
	void operator=(const HalfEdge&) = delete; 
	static HalfEdge* GetInstance();
	void clear();
	void initialiseEdges(TriangleMesh* tm);
	int querySize;
	void calculateOneRing(int vertex);
	void calculateNormals(TriangleMesh* tm, FileManager* fm);
	void fillAdjascencyList(TriangleMesh* tm);
	void bfs(int source, TriangleMesh* tm, FileManager* fm);
};