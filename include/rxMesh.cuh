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

	//cuda pointers.

	int* d_adjascentTriangles;
	int* d_faceVector;
	int* d_triangleAdjascencyVector;
	int* d_sizeN;
	int* d_patchingArray;
	int* d_seedArray;
	//
	int patchSize;
	int patchCount;
	int numFaces;
	//number of seed elements should be the size of patches.
	void h_initialiseSeedElements(TriangleMesh* tm, ComponentManager* cm, int pc);
	void h_initialiseSeedElementsMultiComp(TriangleMesh* tm, ComponentManager* cm);
	void initialise(TriangleMesh* tm);

	void h_fillPatchingArrayWithSeedPoints();

	void h_populatePatches(TriangleMesh* tm);

	//all the host functions that call gpu functions will have h_ tag
	void h_fillAdjascentTriangles(TriangleMesh* tm);
	//fill patching array.
	~preRxMeshDataStructure();
	void freeCudaData();

	void clear();
	void clearSeedComponents(TriangleMesh* tm);


};

//cuda functions.
__global__
void d_fillAdjascentTriangles(int* d_faceVector, int* d_adjascentTriangles, int size_N);

__global__
void d_populatePatchingArray(int* d_patchingArray, int size_N, int* d_adjascentTriangles);

//__global__
//void d_populatePatchingArray(int* d_patchingArray, int size_N, int* d_adjascentTriangles, int* d_count, bool* d_continue);


__global__
void d_counter(int* d_patchingArray, int size_N, int* d_count);