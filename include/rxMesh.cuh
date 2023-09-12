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

	//cuda pointers.

	int* d_adjascentTriangles;
	int* d_faceVector;
	int* d_triangleAdjascencyVector;
	int* d_sizeN;
	//
	int patchSize;
	int patchCount;
	int numFaces;
	//number of seed elements should be the size of patches.
	void h_initialiseSeedElements(TriangleMesh* tm, ComponentManager* cm, int ps);
	std::vector<int> seedElements;
	void initialise(TriangleMesh* tm);

	//all the host functions that call gpu functions will have h_ tag
	void h_fillAdjascentTriangles(TriangleMesh* tm);
	//
	~preRxMeshDataStructure();
	void freeCudaData();

};


__global__
void d_fillAdjascentTriangles(int* d_faceVector, int* d_adjascentTriangles, int size_N);
