#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

#include <iostream>
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/common.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../external/stb_image.h"

#include <string>
#include <thread>
#include <mutex>


#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> 


#include "../include/halfedge.h"



extern std::string objFileName;




class Initialisation
{
	//class is used to read the obj files and manageOperations.
	//to be extended to triangle meshes as well.
	//should be singleton

protected:
	Initialisation();

	static Initialisation* initialise;

public:
	Initialisation(Initialisation& other) = delete;
	void operator=(const Initialisation&) = delete; 
	static Initialisation* GetInstance();
	//void readObjFile(const std::string filePath);
};