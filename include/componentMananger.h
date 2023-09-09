#include "../include/renderer.h"
#include <unordered_set>
#include <algorithm>

class ComponentManager
{
	//class is used to read the obj files and manageOperations.
	//to be extended to triangle meshes as well.
	//should be singleton

protected:
	ComponentManager();
	static ComponentManager* cm;
    std::vector<int> parentMatrix;
    std::vector<int> rankMatrix;
	int vertexCount = 0;
	int componentCount = 0;

public:
	ComponentManager(ComponentManager& other) = delete;
	void operator=(const ComponentManager&) = delete; 
	static ComponentManager* GetInstance();
    void clear();
    void initialiseMatrices(int n);
    int findParent(int v0);
    void unionVertices(int v0, int v1);
	void findComponents(TriangleMesh* tm);
};