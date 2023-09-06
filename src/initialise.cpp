#include "../include/initialise.h"




//file reader class
Initialisation* Initialisation::initialise = nullptr;

Initialisation::Initialisation()
{}


Initialisation* Initialisation::GetInstance()
{
    if (initialise == nullptr) {
        initialise = new Initialisation();
    }
    return initialise;
}

void Initialisation::objFileWindowHandler(bool &showLoaderWindow, FileManager* fm)
{
    if (fm == nullptr)
        return;
    bool loadAgain = false;
    ImVec2 size = ImVec2(200, 100);
    ImVec2 position = ImVec2(800, 0);
    ImGui::SetNextWindowSize(size);
    ImGui::SetNextWindowPos(position);
    ImGui::Begin("Object Loader Window", &showLoaderWindow);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
    ImGui::Text("Load object file Here!");
    if (ImGui::Button("Close"))
        showLoaderWindow = false;
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
                ImGui::End();
            }
            else
                objFileName = objFile;
            if (objFileName.find("obj") == std::string::npos)
            {
                std::cout << "Wrong format selected" << std::endl;
                loadAgain = true;
            }
            showLoaderWindow = loadAgain;
        }
    }
    ImGui::End();
}