#include "../include/triangle.h"

TriangleMesh* TriangleMesh::triangle = nullptr;

TriangleMesh::TriangleMesh()
{}


TriangleMesh* TriangleMesh::GetInstance()
{
    if (triangle == nullptr) {
        triangle = new TriangleMesh();
    }
    return triangle;
}

void TriangleMesh::clear()
{
    vertices.clear();
    normals.clear();
    textures.clear();
    faces.clear();
    faceVector.clear();
}

