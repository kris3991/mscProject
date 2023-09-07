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

void RendererManager::assignRenderModel(ShaderManager* sm, bool enableWireFrame)
{
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, rt.size() * sizeof(rt), &rt[0], GL_STATIC_DRAW);

    //access the data for attributes.
    GLint posAttrib = glGetAttribLocation(sm->shaderProgram, "aPosition");
    glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(posAttrib);

    // normal attribute
    GLint normalAttrib = glGetAttribLocation(sm->shaderProgram, "aNormal");
    glVertexAttribPointer(normalAttrib, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(normalAttrib);


    // texture coord attribute
    GLint textureAttrib = glGetAttribLocation(sm->shaderProgram, "aTexCoord");
    glVertexAttribPointer(textureAttrib, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(textureAttrib);

    //size of the bounding window is calculated in while loop, so cant set projection here. Fix later!

    if(enableWireFrame)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

}

void RendererManager::drawModel(ShaderManager* sm)
{
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(sm->shaderProgram);
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::scale(model, glm::vec3(0.25f, 0.25f, 0.25f));
    //model = glm::rotate(model, (float)glfwGetTime(), glm::vec3(0.0f, 1.0f, 0.0f));
    GLuint modelLoc = glGetUniformLocation(sm->shaderProgram, "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

    glm::mat4 view = glm::mat4(1.0f);
    view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
    GLuint viewLoc = glGetUniformLocation(sm->shaderProgram, "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

    glm::mat4 projection;
    projection = glm::perspective(glm::radians(45.0f), (float)1000 / (float)800, 0.1f, 100.0f);
    GLuint projectionLoc = glGetUniformLocation(sm->shaderProgram, "projection");
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));


    glBindVertexArray(VAO);
    //glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glDrawArrays(GL_TRIANGLES, 0, rt.size());
}

void RendererManager::clearBuffer()
{
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}