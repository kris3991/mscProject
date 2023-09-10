#include "../include/fileManager.h"

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
			ss >> f.v0 >> slash >> f.t0 >> slash >> f.n0;
			ss >> f.v1 >> slash >> f.t1 >> slash >> f.n1;
			ss >> f.v2 >> slash >> f.t2 >> slash >> f.n2;
			// Convert to 0-based index
			f.v0--; f.t0--; f.n0--;
			f.v1--; f.t1--; f.n1--;
			f.v2--; f.t2--; f.n2--;
			

			//for rendering.
			tm->faces.push_back(f);

			//basically the face data structure.
			tm->faceVector.push_back(f.v0);
			tm->faceVector.push_back(f.v1);
			tm->faceVector.push_back(f.v2);

		}
		++count;
	}
	return;
}

const char* FileManager::launchFileReader()
{
    return (tinyfd_openFileDialog ( "Select Object file to Load",	NULL,	0, NULL, NULL, 1));
}

void FileManager::writeBFSdata(TriangleMesh* tm, std::map<int, int> &geodesicDistance, std::chrono::duration<double> geodesicProcessingTime,
	std::chrono::duration<double> geodesicCalcTime, int source)
{
	std::string fileName = "geodesic.txt";
	std::ofstream bfsFile(fileName);
        if (!bfsFile.is_open())
        {
            std::cerr << "Can't open file: " << fileName << std::endl;
            return;
        }
        else
        {
			bfsFile << "Geodesic distance"  << std::endl;
            bfsFile << "Initial Processing time:\t"<< geodesicProcessingTime << std::endl;
            bfsFile << "calculation time:\t" << geodesicCalcTime << std::endl;
            bfsFile << "Number of vertices:\t" << tm->vertices.size() << std::endl;
            bfsFile << "Number of faces:\t" << tm->faceVector.size() << std::endl;
            bfsFile << "Source " << source << std::endl;
            for (int i = 0; i < geodesicDistance.size(); ++i)
            {
                if (geodesicDistance[i] != std::numeric_limits<int>::max())
                    bfsFile << "vertex: " << i << "\t" << geodesicDistance[i] << std::endl;
                else
                    bfsFile << "vertex: " << i << "\t" << "Disjoint component" << std::endl;
            }
            bfsFile.close();
		}
	
}

void  FileManager::writeNormals(TriangleMesh* tm, std::chrono::duration<double> delta)
{
	std::string fileName = "normals.txt";
	std::ofstream normalFile(fileName);
	if (!normalFile.is_open())
	{
		std::cerr << "Can't open file: " << fileName << std::endl;
		return;
	}
	else
	{
		normalFile << "Number of vertices: " << tm->vertices.size() << std::endl;
		normalFile << "Number of faces: " << tm->faceVector.size() << std::endl;
		normalFile << "time taken: " << std::endl;
		normalFile << delta << std::endl;
		for (int i = 0; i < tm->normals.size(); ++i)
		{
			normalFile << tm->normals[i].x << "\t" << tm->normals[i].y << "\t" << tm->normals[i].z << std::endl;
		}
		normalFile.close();
		std::cout << "normal file written" << std::endl;
	}
}