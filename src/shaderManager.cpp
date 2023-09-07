#include "../include/shaderManager.h"

ShaderManager* ShaderManager::sm = nullptr;

ShaderManager::ShaderManager()
{}


ShaderManager* ShaderManager::GetInstance()
{
    if (sm == nullptr) {
        sm = new ShaderManager();
    }
    return sm;
}

void ShaderManager::readShader(std::string filePath, std::string& shaderSrc)
{
    std::string output;
    if (!filePath.size())
    {
        std::cout << "empty file path" << std::endl;
        return;
    }
    std::ifstream input(filePath.c_str());
    std::stringstream s;
    s << input.rdbuf();
    output = s.str();
    if (output.size())
    {
        shaderSrc = output;
    }
    return;
}

void ShaderManager::printLogs(GLuint prog, GLenum pname, std::string msg)
{
    GLint status;
    char log[512];
    glGetProgramiv(prog, pname, &status);
    if (!status) {
        glGetProgramInfoLog(prog, 512, NULL, log);
        std::string output = "ERROR::SHADER::PROGRAM::" + msg + "\n";
        std::cout << output << log << std::endl;
    }
}

void ShaderManager::compileShaders(GLuint &shader, const char* shaderSrc, GLenum shaderType)
{
    shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, &shaderSrc, NULL);
    glCompileShader(shader);
    printLogs(shader, GL_COMPILE_STATUS, "SHADER::COMPILATION_FAILED");
}

void ShaderManager::createShaderProgram()
{
    GLenum err = glewInit();
    GLuint vertexShader{ 0 }, fragmentShader{0};
    const char* vertexShaderSource = 0, * fragmentShaderSource = 0;
    std::string v;
    std::string vertexShaderPath = "../resources/vertex.glsl";
    readShader(vertexShaderPath, v);
    vertexShaderSource = v.c_str();
    compileShaders(vertexShader, vertexShaderSource, GL_VERTEX_SHADER);
    std::string fragmentShaderPath = "../resources/fragment.glsl";
    readShader(fragmentShaderPath, v);
    fragmentShaderSource = v.c_str();
    compileShaders(fragmentShader, fragmentShaderSource, GL_FRAGMENT_SHADER);

    std::vector<GLuint> shaders;
    shaders.push_back(vertexShader);
    shaders.push_back(fragmentShader);



    shaderProgram = glCreateProgram();
    for (int i = 0; i < shaders.size(); ++i)
        glAttachShader(shaderProgram, shaders[i]);
    glLinkProgram(shaderProgram);
    printLogs(shaderProgram, GL_LINK_STATUS, "LINKING_FAILED");
    /*int status;
    char log[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
    if (!status) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, log);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << log << std::endl;
    }*/
    for (int i = 0; i < shaders.size(); ++i)
        glDeleteShader(shaders[i]);
}

void ShaderManager::clear()
{
    this->shaderProgram = 0;
}