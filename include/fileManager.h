
#include <vector>
#include <string>
#include <algorithm>


//glm
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


//obj files.
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <fstream>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

//tiny
#include "../external/tinyfiledialogs/tinyfiledialogs.h"

#include "../include/triangle.h"

class FileManager
{
	//class is used to read the obj files and manageOperations.
	//to be extended to triangle meshes as well.
	//should be singleton

protected:
	FileManager();

	static FileManager* fm;

public:
	FileManager(FileManager& other) = delete;
	void operator=(const FileManager&) = delete;
	static FileManager* GetInstance();
	void readObjFile(const std::string filePath, TriangleMesh* tm);
	const char* launchFileReader();
};