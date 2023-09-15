//#include "../include/componentMananger.cuh"
#include "../include/halfedge.h"

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "../include/test.cuh"
#include <random>
#include <stdio.h>

struct Vertex
{
	float vx;
	float vy; 
	float vz;
};

struct Faces
{
	//i dont really want to add 12 bytes more by adding normals here.
	int v0;
	int v1;
	int v2;

	float v0PosX;
	float v0PosY;
	float v0PosZ;

	float v1PosX;
	float v1PosY;
	float v1PosZ;

	float v2PosX;
	float v2PosY;
	float v2PosZ;


};

class preRxMeshDataStructure;

class Patch
{
public:
	std::set<int> patchWithRings;
	int* d_patchWithRings;
	//not really needed but just in case.
	std::vector<int> boundaryElements;
	std::set<int> ribbonElements;
	std::vector<int> faces;
	int newPatchSize;
	void fillRibbons(preRxMeshDataStructure* rm);
};

class RxMesh
{
protected:
	RxMesh();
	static RxMesh* rxMesh;
public:

	RxMesh(RxMesh& other) = delete;
	void operator=(const RxMesh&) = delete;
	static RxMesh* GetInstance();
	std::vector<Vertex> vertices;
	std::vector<Faces> faces;
	std::vector<Vertex> normals;
	std::vector<Patch> patches;
	int* d_globalPatch;
	int patchCount;
	int patchSize;
	void calculateNormals(FileManager* fm, TriangleMesh* tm);
	void clearCudaData();

	//cuda data.
	Faces* d_faces = 0;
	Vertex* d_normals = 0;

};



class preRxMeshDataStructure
{
protected:
	preRxMeshDataStructure();
	static preRxMeshDataStructure* rxMeshStruct;
public:

	preRxMeshDataStructure(preRxMeshDataStructure& other) = delete;
	void operator=(const preRxMeshDataStructure&) = delete;
	static preRxMeshDataStructure* GetInstance();


	std::vector<int> h_adjascentTriangles;
	std::vector<int> h_seedElements;
	//for patching.
	std::vector<int> h_faceIndexVector;
	int sizeofFaceVector;
	//for multi component meshes.
	std::vector<int> multiComponentPatchCount;
	std::vector<int> multiComponentPatchSize;
	std::vector<int> h_patchingArray;
	std::vector<int> h_tempPatchArray;
	

	std::map<int, std::vector<int>> h_adjTriMap;
	std::vector<int> h_boundaryElements;

	//cuda pointers.

	int* d_adjascentTriangles = 0;
	int* d_faceVector = 0;
	int* d_triangleAdjascencyVector = 0;
	int* d_sizeN = 0;
	int* d_patchingArray = 0;
	int* d_seedArray = 0;
	int* d_boundaryElements = 0;
	int* d_patchPositions = 0;
	//
	int patchSize;
	int patchCount;
	int numFaces;
	//number of seed elements should be the size of patches.
	void h_initialiseSeedElements(TriangleMesh* tm, ComponentManager* cm, int pc);
	void h_initialiseSeedElementsMultiComp(TriangleMesh* tm, ComponentManager* cm);
	void initialise(TriangleMesh* tm);

	void h_fillPatchingArrayWithSeedPoints();

	void h_populatePatches(TriangleMesh* tm, bool doIterations, ComponentManager* cm, int pc);

	//all the host functions that call gpu functions will have h_ tag
	void h_fillAdjascentTriangles(TriangleMesh* tm);
	//fill patching array.
	~preRxMeshDataStructure();
	void freeCudaData();

	void clear();
	void clearSeedComponents(TriangleMesh* tm);

	void addRibbons(TriangleMesh* tm, RxMesh* rMesh);

	void fillVertices(RxMesh* rm, TriangleMesh* tm);


};

//cuda functions.
__global__
void d_fillAdjascentTriangles(int* d_faceVector, int* d_adjascentTriangles, int size_N);

__global__
void d_populatePatchingArray(int* d_patchingArray, int size_N, int* d_adjascentTriangles);




__global__
void d_counter(int* d_patchingArray, int size_N, int* d_count);

__global__ 
void d_computePatchCount(int* d_patchingArray, int* d_individualCounts, int size_N);


__global__
void d_arrangePatches(int* d_patchingArray, int* d_newPatchArray, 
	int* d_individualCounts, int* d_prefixSum, int size_N);

__global__
void d_findBoundaryPoints(int* d_patchingArray, int size_N, int* d_boundaryElements, int* d_adjascentTriangles, int* d_patchPositions);

__global__
void d_calculateNormals(Faces* d_faces, int size_N, Vertex* d_normals, int normalCount);




