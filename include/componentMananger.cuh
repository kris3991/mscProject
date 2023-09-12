#include "../include/renderer.h"
#include <unordered_set>
#include <algorithm>

#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>

class ComponentManager
{
	//class is used to read the obj files and manageOperations.
	//to be extended to triangle meshes as well.
	//should be singleton

protected:
	ComponentManager();
	static ComponentManager* cm;
	std::map<int, int> parentMap;
	std::map<int, int> rankMap;
	

public:

	int vertexCount = 0;
	int componentCount = 0;
	std::vector<int> componentLocation;

	ComponentManager(ComponentManager& other) = delete;
	void operator=(const ComponentManager&) = delete; 
	static ComponentManager* GetInstance();
    void clear();
    void initialiseMatrices(int n);
    int findParent(int v0);
    void unionVertices(int v0, int v1);
	void findComponents(TriangleMesh* tm);
};