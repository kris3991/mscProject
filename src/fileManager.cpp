#include "../include/fileManager.h"
#include "../external/tinyfiledialogs/tinyfiledialogs.h"
//file reader class
FileManager* FileManager::fm = nullptr;

FileManager::FileManager()
{}


FileManager* FileManager::GetInstance()
{
    if (fm == nullptr) {
        fm = new FileManager();
    }
    return fm;
}

void FileManager::readObjFile(const std::string filePath, TriangleMesh* tm)
{
    if (tm == nullptr || filePath.size() == 0)
    {
        std::cout << "invalid file read!" << std::endl;
        return;
    }
	tm->clear();
	std::string line;
	std::ifstream objFile(filePath);
	int count = 0;
	while (std::getline(objFile, line))
	{
		std::istringstream ss(line);
		std::string attribute;
		ss >> attribute;
		if (attribute == "v")
		{
			glm::vec3 vertex;
			ss >> vertex.x >> vertex.y >> vertex.z;
			tm->vertices.push_back(vertex);
		}
		else if (attribute == "vt")
		{
			glm::vec2 texture;
			ss >> texture.x >> texture.y;
			tm->textures.push_back(texture);
		}
		else if (attribute == "vn")
		{
			glm::vec3 normal;
			ss >> normal.x >> normal.y >> normal.z;
			tm->normals.push_back(normal);
		}
		else if (attribute == "f")
		{
			face f;
			char slash;
			ss >> f.v1 >> slash >> f.t1 >> slash >> f.n1;
			ss >> f.v2 >> slash >> f.t2 >> slash >> f.n2;
			ss >> f.v3 >> slash >> f.t3 >> slash >> f.n3;
			// Convert to 0-based index
			f.v1--; f.t1--; f.n1--;
			f.v2--; f.t2--; f.n2--;
			f.v3--; f.t3--; f.n3--;
			tm->faces.push_back(f);
		}
		++count;
	}
	return;
}

const char* FileManager::launchFileReader()
{
    return (tinyfd_openFileDialog ( "Select Object file to Load",	NULL,	0, NULL, NULL, 1));
}