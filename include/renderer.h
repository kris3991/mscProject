#include "../include/fileManager.h"

struct RenderTriangles {
    float v0, v1, v2;      // Vertex coordinates
    float n0, n1, n2;   // Normal coordinates
    float s, t;         // Texture coordinates
};

class RendererManager
{
	//class is used to read the obj files and manageOperations.
	//to be extended to triangle meshes as well.
	//should be singleton

protected:
	RendererManager();

	static RendererManager* rm;

public:

    std::vector<RenderTriangles> rt;
    
	RendererManager(RendererManager& other) = delete;
	void operator=(const RendererManager&) = delete; 
	static RendererManager* GetInstance();
	void clear();
	void fillRenderTriangles(TriangleMesh* tm);
};