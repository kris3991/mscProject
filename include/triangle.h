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

struct Edge
{
	int v0;
	int v1;
};

class TriangleMesh
{
	//takes in the data read from fileManagement class.

protected:
	TriangleMesh();

	static TriangleMesh* triangle;

public:

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> textures;
	//the following face is for reading the values from the obj file. has vertices, normals and textures.
    std::vector<face> faces;
	//has the face information ie v0, v1, v2.. Dont mix.
	std::vector<int>faceVector;
    
	TriangleMesh(TriangleMesh& other) = delete;
	void operator=(const TriangleMesh&) = delete; 
	static TriangleMesh* GetInstance();
	void clear();
	
};