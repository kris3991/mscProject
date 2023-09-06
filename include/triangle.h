#include <vector>
#include <string>
#include <GL/glew.h>
#include <glm/common.hpp>
#include <glm/gtc/type_ptr.hpp>


struct face
{
    GLuint v0, v1, v2;     // Vertex indices
    GLuint n0, n1, n2;     // Normal indices
    GLuint t0, t1, t2;     // Texture coordinate indices //just for symmetry :D
};

class TriangleMesh
{
	//class is used to read the obj files and manageOperations.
	//to be extended to triangle meshes as well.
	//should be singleton

protected:
	TriangleMesh();

	static TriangleMesh* triangle;

public:

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> textures;
    std::vector<face> faces;
    
	TriangleMesh(TriangleMesh& other) = delete;
	void operator=(const TriangleMesh&) = delete; 
	static TriangleMesh* GetInstance();
	void clear();
};