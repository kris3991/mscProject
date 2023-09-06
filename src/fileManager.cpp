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

void FileManager::readObjFile(const std::string filePath)
{

}

const char* FileManager::launchFileReader()
{
    return (tinyfd_openFileDialog ( "Select Object file to Load",	NULL,	0, NULL, NULL, 1));
}