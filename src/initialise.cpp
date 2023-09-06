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
