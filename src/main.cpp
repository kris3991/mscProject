
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

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        //size management
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        //launch obj file loader.
        

        if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_L)))
        {
            showLoaderWindow = true;
        }


        if (showLoaderWindow)
        {
           //object loading window.
            bool loadAgain = false;
            ImVec2 size = ImVec2(200, 100);
            
            ImVec2 position = ImVec2(display_w - display_w /6, display_h - (display_h - display_h/20));
            ImGui::SetNextWindowSize(size);
            ImGui::SetNextWindowPos(position);
            ImGui::Begin("Object Loader Window", &showLoaderWindow);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Load object file Here!");
            if (ImGui::Button("Close"))
            {
                showLoaderWindow = false;

            }
            if (showLoaderWindow && ImGui::Button("Load Object file!"))
            {
                std::cout << "file launcher launched" << std::endl;
                if (fm != nullptr)
                {
                    const char* objFile = fm->launchFileReader();
                    if (objFile == nullptr)
                    {
                        ImGui::Begin("Object loading failed", &showLoaderWindow);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
                        ImGui::Text("Select Another Window!");
                        if (ImGui::Button("Close"));
                        loadAgain = true;
                        
                    }
                    else
                        objFileName = objFile;
                    if (objFileName.find("obj") == std::string::npos)
                    {
                        std::cout << "Wrong format selected" << std::endl;
                        loadAgain = true;
                    }
                    else
                        processObjFile = true;
                    showLoaderWindow = loadAgain;

                }
            }
            ImGui::End();
        }

        if (processObjFile)
        {
            fm->readObjFile(objFileName, tm);
            processObjFile = false;
            rm->fillRenderTriangles(tm);
        }
       


        // Rendering
        ImGui::Render();
       
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }
#ifdef __EMSCRIPTEN__
    EMSCRIPTEN_MAINLOOP_END;
#endif

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}