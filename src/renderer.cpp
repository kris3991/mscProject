#include "../include/renderer.h"

RendererManager* RendererManager::rm = nullptr;

RendererManager::RendererManager()
{}


RendererManager* RendererManager::GetInstance()
{
    if (rm == nullptr) {
        rm = new RendererManager();
    }
    return rm;
}

void RendererManager::clear()
{
    rt.clear();
}

void RendererManager::fillRenderTriangles(TriangleMesh* tm)
{
    //for the vbo we are using the format v0, v1, v2, n0, n1, n2, t0, t1
    if (tm == nullptr)
    {
        std::cout << "triangle mesh is empty" << std::endl;
        return;
    }
    rt.clear();
    for (const auto &triangle : tm->faces)
    {
        //vertex 0
        rt.push_back({ tm->vertices[triangle.v0].x, tm->vertices[triangle.v0].y, tm->vertices[triangle.v0].z,
                       tm->normals[triangle.n0].x, tm->normals[triangle.n0].y, tm->normals[triangle.n0].z,
                       tm->textures[triangle.t0].x, tm->textures[triangle.t0].y });

        //vertex 1
        rt.push_back({ tm->vertices[triangle.v1].x, tm->vertices[triangle.v1].y, tm->vertices[triangle.v1].z, 
                       tm->normals[triangle.n1].x, tm->normals[triangle.n1].y, tm->normals[triangle.n1].z, 
                       tm->textures[triangle.t1].x, tm->textures[triangle.t1].y });

        //vertex 2
        rt.push_back({ tm->vertices[triangle.v2].x, tm->vertices[triangle.v2].y, tm->vertices[triangle.v2].z,
                       tm->normals[triangle.n2].x, tm->normals[triangle.n2].y, tm->normals[triangle.n2].z,
                       tm->textures[triangle.t2].x, tm->textures[triangle.t2].y });
    }
    int a = 30;
}