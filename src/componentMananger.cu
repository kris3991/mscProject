#include "../include/componentMananger.cuh"

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
    parentMap.clear();
    rankMap.clear();
}

void ComponentManager::initialiseMatrices(int n)
{
    vertexCount = n;
    //initialise all ranks as zero and parentMatrix as itself.
    for(int i = 0; i < n; ++i)
    {
        parentMap[i] = i;
        rankMap[i] = 0;
    }
}



int ComponentManager::findParent(int v0)
{
    //path compression
    //keep moving up.
    //basically to the root.
    /*if(parentMatrix[v0] != v0)
        parentMatrix[v0] = findParent(parentMatrix[v0]);
    return parentMatrix[v0];*/
    if (parentMap[v0] != v0)
        parentMap[v0] = findParent(parentMap[v0]);
    return parentMap[v0];

}


void ComponentManager::unionVertices(int v0, int v1)
{
    int pV0 = findParent(v0);
    int pV1 = findParent(v1);
    //union find: referred from geekforgeeks : https://www.geeksforgeeks.org/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/
    if (pV0 != pV1) 
    {
        //if (rankMatrix[pV0] < rankMatrix[pV1]) {
        //    parentMatrix[pV0] = pV1;
        //} else if (rankMatrix[pV0] > rankMatrix[pV1]) {
        //    parentMatrix[pV1] = pV0;
        //} else {
        //    //this is random.
        //    parentMatrix[pV1] = pV0;
        //    rankMatrix[pV0]++;
        //}
        if (rankMap[pV0] < rankMap[pV1]) {
            parentMap[pV0] = pV1;
        }
        else if (rankMap[pV0] > rankMap[pV1]) {
            parentMap[pV1] = pV0;
        }
        else {
            //this is random.
            parentMap[pV1] = pV0;
            rankMap[pV0]++;
        }
        
       


    }
}



void ComponentManager::findComponents(TriangleMesh* tm)
{
    componentCount = 0;
    componentLocation.clear();
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

    std::vector<int> parentVector(tm->faceVector.size());

    for (int i = 0; i < tm->faceVector.size(); ++i)
    {
        parentVector[i] = parentMap[tm->faceVector[i]];
    }
    //sometimes with blender files subdivision causes some issues.
    //i am generating large face objects with subdivision,
    //this is an extra check.
    //each face has 3 vertices, in a face it is impossible to have vertices that have distinct parents, check that.
    //faces belonging to the same group will be grouped together.
    for (int i = 0; i < parentVector.size(); i = i+3)
    {
        int v0, v1, v2;
        v0 = parentVector[i];
        v1 = parentVector[i + 1];
        v2 = parentVector[i + 2];
        if (!(v0 == v1 && v1 == v2))
        {
            parentVector[i + 1] = parentVector[i];
            parentVector[i + 2] = parentVector[i + 1];
            //check with previous face just to be sure.
            if (parentVector[i - 1] != parentVector[i])
            {
                parentVector[i] = parentVector[i - 1];
                parentVector[i + 1] = parentVector[i];
                parentVector[i + 2] = parentVector[i + 1];
            }
            
        }
    }


   //find where each component begins and ends.
    componentLocation.push_back(0);
    for (int i = 1; i < parentVector.size(); ++i)
    {
        //we get the face location by dividing by 3.
        if (parentVector[i - 1] != parentVector[i])
            componentLocation.push_back(i/3);
    }
    
    componentCount = componentLocation.size();
    componentLocation.push_back((tm->faceVector.size() / 3));
   
}