#include <vector>
#include <string>
#include <GL/glew.h>
#include <glm/common.hpp>
#include <glm/gtc/type_ptr.hpp>


struct face
{
    GLuint v1, v2, v3;     // Vertex indices
    GLuint n1, n2, n3;     // Normal indices
    GLuint t1, t2, t3;     // Texture coordinate indices
};

// class TriangleMesh
// {
//     public:
        // std::vector<glm::vec3> vertices;
        // std::vector<glm::vec3> normals;
        // std::vector<glm::vec2> textures;
        // std::vector<face> faces;

//         // TriangleMesh(const std::vector<glm::vec3> &v, std::vector<glm::vec3> &n, std::vector<glm::vec2> &t, std::vector<face> &f):
//         //     vertices(v), normals(n), textures(t), faces(f) {};
//         // void clear();    
// };

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