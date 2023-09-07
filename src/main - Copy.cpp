
#include "../include/test.cuh"
#include "../include/initialise.h"


#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

//glfw callbacks
static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

//global file
std::string objFileName = std::string();

struct GLVertex2
{
    glm::vec3 vertex;
};


// Main code
int main(int argc, char** argv)
{
    GLFWwindow* window = nullptr;
    ImVec2 windowSize;

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif


    //initialise all the singletons.
    Initialisation* initManager = Initialisation::GetInstance();
    if (initManager == nullptr)
    {
        std::cout << "initialisation failed" << std::endl;
        return 0;
    }

    FileManager* fm = FileManager::GetInstance();
    if(fm == nullptr)
    {
        std::cout << "file manager initialisation failed" << std::endl;
        return 0;
    }

    TriangleMesh* tm = TriangleMesh::GetInstance();
    if(tm == nullptr)
    {
        std::cout << "triangle mesh uninitialised" << std::endl;
        return 0;
    }

    RendererManager* rm = RendererManager::GetInstance();
    if(rm == nullptr)
    {
        std::cout << "Render Manager uninitialised" << std::endl;
        return 0;
    }

    ShaderManager* sm = ShaderManager::GetInstance();
    if(sm == nullptr)
    {
        std::cout << "Shader Manager uninitialised" << std::endl;
        return 0;
    }
    
    window = glfwCreateWindow(1000, 800, "MSc Project", nullptr, nullptr);
    

    //imgui window management. From the example in imgui git.

    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    //initiliase tiny dialog to open dialog
    //fm->launchFileReader();


    //imgui window management. end

    // Our state
    bool showLoaderWindow = true;
    bool processObjFile = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);



    int n = 1000;
    int* c_i = new int[n];
    test(c_i, n);

    //// Output the result
    //for (int i = 0; i < n; i++) {
    //    std::cout << "c[" << i << "] = " << c_i[i] << std::endl;
    //}

    delete[] c_i;
    bool isObjectLoaded = false;
    std::string fileName;

    //initialise glew
    GLenum err = glewInit();
    if (err)
    {
        std::cout << "GLEW INITIALISATION FAILED " << err << std::endl;
    }
    glEnable(GL_DEPTH_TEST);

    //load shaders.
    //load shaders.


    
    

    fm->readObjFile("C:/Users/gopik/OneDrive/Desktop/gitCheckout/mscProject/Models/sphere.obj", tm);
    sm->createShaderProgram();
    rm->fillRenderTriangles(tm);
    rm->assignRenderModel(sm, true);
    


    while (!glfwWindowShouldClose(window))
    {
        if (1)
        {
            rm->drawModel(sm);
        }
        else
        {
            rm->clearBuffer();
        }

        //rendering end
        glfwPollEvents();
        glfwSwapBuffers(window);
    }


    //// Cleanup
    //ImGui_ImplOpenGL3_Shutdown();
    //ImGui_ImplGlfw_Shutdown();
    //ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}