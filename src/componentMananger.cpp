#include "../include/componentMananger.h"

ComponentManager* ComponentManager::cm = nullptr;

ComponentManager::ComponentManager()
{}


ComponentManager* ComponentManager::GetInstance()
{
    if (cm == nullptr) {
        cm = new ComponentManager();
    }
    return cm;
}

void ComponentManager::clear()
{
    componentCount = 0;
    vertexCount = 0;
    rankMatrix.clear();
    parentMatrix.clear();
}

void ComponentManager::initialiseMatrices(int n)
{
    vertexCount = n;
    rankMatrix.resize(n);
    parentMatrix.resize(n);
    //initialise all ranks as zero and parentMatrix as itself.
    for(int i = 0; i < n; ++i)
    {
        rankMatrix[i] = 0;
        parentMatrix[i] = i;
    }
}



int ComponentManager::findParent(int v0)
{
    //path compression
    //keep moving up.
    //basically to the root.
    if(parentMatrix[v0] != v0)
        parentMatrix[v0] = findParent(parentMatrix[v0]);
    return parentMatrix[v0];
}


void ComponentManager::unionVertices(int v0, int v1)
{
    int pV0 = findParent(v0);
    int pV1 = findParent(v1);
    //union find: referred from geekforgeeks : https://www.geeksforgeeks.org/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/
    if (pV0 != pV1) 
    {
        if (rankMatrix[pV0] < rankMatrix[pV1]) {
            parentMatrix[pV0] = pV1;
        } else if (rankMatrix[pV0] > rankMatrix[pV1]) {
            parentMatrix[pV1] = pV0;
        } else {
            //this is random.
            parentMatrix[pV1] = pV0;
            rankMatrix[pV0]++;
        }
    }
}

//void ComponentManager::findComponents(TriangleMesh* tm)
//{
//    for (const auto& edge : tm->edges) {
//        int v0 = edge.v0;
//        int v1 = edge.v1;
//        int p0 = findParent(v0);
//        int p1 = findParent(v1);
//
//        if (p0 != p1) {
//            unionVertices(p0, p1);
//        }
//    }
//
//    std::unordered_set<int> components;
//    for (int i = 0; i <vertexCount; ++i) {
//        components.insert(findParent(i));
//    }
//
//    int a = 30;
//
//}

void ComponentManager::findComponents(TriangleMesh* tm)
{
    for (int i = 0; i < tm->faceVector.size(); ++i)
    {
        int v0 = tm->faceVector[i], v1;
        if (i % 3 == 2)
            v1 = tm->faceVector[i - 2];
        else
            v1 = tm->faceVector[i+1];

        int pv0 = findParent(v0);
        int pv1 = findParent(v1);

        if(pv0 != pv1)
            unionVertices(pv0, pv1);

    }

    //all the vertices that have commone root is one component
    //initialise
    //we will always atleast one component.
    //this is o(n2) but what are the chances of 1000 component object?
    //we need the vertex information during patching.
    //test in cuda.
    componentCount = 1;
    //maintain the distinct arrays.
    std::vector<int> parVector;
    parVector.push_back(findParent(0));
   

    std::vector<std::vector<int>> components;
    for (int i = 0; i < vertexCount; ++i)
    {
        int par = findParent(i);
        if (std::find(parVector.begin(), parVector.end(), par) == parVector.end())
        {
            parVector.push_back(par);
            componentCount++;
        }
    }
    


   /* std::unordered_set<int> components;
    for (int i = 0; i < vertexCount; ++i) {
        components.insert(findParent(i));
    }*/

    int a = 30;

}