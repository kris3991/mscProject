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

void FileManager::readObjFile(const std::string filePath)
{

}