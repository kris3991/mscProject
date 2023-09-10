#include "../include/halfEdge.h"


HalfEdge* HalfEdge::he = nullptr;

HalfEdge::HalfEdge()
{}


HalfEdge* HalfEdge::GetInstance()
{
    if (he == nullptr) {
        he = new HalfEdge();
    }
    return he;
}

void HalfEdge::clear()
{
    edgeStructure.clear();
    edgeStructure2.clear();
    hES.clear();
   
}

void HalfEdge::initialiseEdges(TriangleMesh* tm)
{
    //fill the edge ds.
    for (int i = 0; i < tm->faceVector.size(); i++)
    {
        int v0 = tm->faceVector[i];
        int v1;
        if (i % 3 == 2)
        {
            v1 = tm->faceVector[i -2];
        }
        else
        {
            v1 = tm->faceVector[i + 1];
        }
        //storing both data as map for o(1).
        std::pair<int, int> edge = std::make_pair(v0, v1);
        edgeStructure[edge] = i;
        edgeStructure2[i] = edge;
    }

    //fill half edge ds
    HalfEdgeStruct he;
    he.twin = -1;
    for (int i = 0; i < tm->faceVector.size(); ++i)
    {
        he.face = i / 3;
        int v0 = tm->faceVector[i];
        int v1 = -1;
        int p1 = -1;
        //next.
        if (i % 3 == 2)
        {
            v1 = tm->faceVector[i - 2];
            he.next = i - 2;
        }
        else
        {
            v1 = tm->faceVector[i + 1];
            he.next = i + 1;
        }
        //previous
        if (i % 3 == 0)
        {
            p1 = tm->faceVector[i + 2];
            he.previous = i + 2;
        }
        else
        {
            p1 = tm->faceVector[i - 1];
            he.previous = i - 1;
        }

        //it works. figure out the weird behavior later.
        std::pair<int, int> edge = std::make_pair(v1, v0);
        he.twin = edgeStructure[edge];
        he.vertex = v0;
        hES[i] = he;

    }

    querySize = tm->vertices.size();

    //for faster one ring calculation.
    HalfEdgeStruct start = hES[0];
    int next = hES[0].next;
    
    for (auto const& it : hES)
    {
        int vertex = it.second.vertex;
        int key = it.first;
        vertexHes[vertex].push_back(key);
    }
    
    //maintaining a hash map for normal calculation for o(1) access.
    for (int i = 0; i < tm->faceVector.size(); i += 3)
    {
        normalsCalculated[i / 3] = false;
       
    }
    int a = 30;
}

void HalfEdge::calculateOneRing(int vertex)
{
    std::set<int> oneRing;
    int size = vertexHes[vertex].size();
    std::cout << "the vertex: " << vertex << " is in" << size << " faces" << std::endl;
    const auto startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; ++i)
    {
        int key = vertexHes[vertex][i];
        int previous = hES[key].previous;
        int next = hES[key].next;
        int pval = hES[previous].vertex;
        int nval = hES[next].vertex;
        oneRing.insert(pval);
        oneRing.insert(nval);
    }
    const auto endTime = std::chrono::high_resolution_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    std::cout << "time taken: " << delta << std::endl;
    std::cout << "vertices are" << std::endl;
    for (std::set<int>::iterator itr = oneRing.begin();
        itr != oneRing.end(); itr++)
    {
        std::cout << *itr << "\t";
    }
    
}



void HalfEdge::calculateNormals(TriangleMesh* tm, std::string fileName)
{
	if (!fileName.size())
		std::cout << "invlid fileName" << std::endl;
	else
	{
		std::ofstream normalFile(fileName);
		if (!normalFile.is_open())
		{
			std::cerr << "Can't open file: " << fileName << std::endl;
			return;
		}
		else
		{

			//for cases where there are less number of normals.
			if (tm->normals.size() < tm->vertices.size())
			{
				tm->normals.resize(tm->vertices.size());
			}
            const auto startTime = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < tm->faceVector.size(); i += 3)
			{
				int currVertex = tm->faceVector[i];
				int nextVertex = tm->faceVector[i + 1];
				int previousVertex = tm->faceVector[i + 2];

				const glm::vec3 currVertexPos = tm->vertices[currVertex];
				const glm::vec3 nextVertexPos = tm->vertices[nextVertex];
				const glm::vec3 prevVertexPos = tm->vertices[previousVertex];

				const glm::vec3 vec1 = nextVertexPos - currVertexPos;
				const glm::vec3 vec2 = prevVertexPos - currVertexPos;

				//fast normal calculation.
				glm::vec3 faceNormal = glm::cross(vec1, vec2);

				tm->normals[currVertex] += faceNormal;
				tm->normals[nextVertex] += faceNormal;
				tm->normals[previousVertex] += faceNormal;

                
			}
            const auto endTime = std::chrono::high_resolution_clock::now();
            auto delta = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
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
}

void HalfEdge::fillAdjascencyList(TriangleMesh* tm)
{
    const auto startTime = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < tm->faceVector.size(); i+=3)
    {
        int v0 = tm->faceVector[i];
        int v1 = tm->faceVector[i+1];
        int v2 = tm->faceVector[i+2];

        //to avoid duplicates we use set.
        adjascencyList[v0].insert(v1);
        adjascencyList[v0].insert(v2);

        adjascencyList[v1].insert(v0);
        adjascencyList[v1].insert(v2);

        adjascencyList[v2].insert(v0);
        adjascencyList[v2].insert(v1);

        visited[v0] = false;
        visited[v1] = false;
        visited[v2] = false;

        //distance
        //for disconnected components.
        geodesicDistance[v0] = std::numeric_limits<int>::max();
        geodesicDistance[v1] = std::numeric_limits<int>::max();
        geodesicDistance[v2] = std::numeric_limits<int>::max();
    }
    const auto endTime = std::chrono::high_resolution_clock::now();
   geodesicProcessingTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

}

void HalfEdge::bfs(int source, std::string fileName, TriangleMesh* tm)
{
    //basic bfs implementation, referred from my old code and geek for geek,
    if (!fileName.size())
        std::cout << "invlid fileName" << std::endl;
    else
    {
        std::ofstream bfsFile(fileName);
        if (!bfsFile.is_open())
        {
            std::cerr << "Can't open file: " << fileName << std::endl;
            return;
        }
        else
        {
            bfsQueue.push(source);
            visited[source] = true;
            geodesicDistance[source] = 0;
            const auto startTime = std::chrono::high_resolution_clock::now();
            while (!bfsQueue.empty())
            {
                int firstElement = bfsQueue.front();
                bfsQueue.pop();

                for (std::set<int>::iterator itr = adjascencyList[firstElement].begin(); itr != adjascencyList[firstElement].end(); itr++)
                {
                    int val = *itr;
                    if (!visited[val])
                    {
                        visited[val] = true;
                        geodesicDistance[val] = geodesicDistance[firstElement] + 1;
                        bfsQueue.push(val);
                    }

                }

            }
            const auto endTime = std::chrono::high_resolution_clock::now();
            geodesicCalcTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
            bfsFile << "Geodesic distance" << tm->faceVector.size() << std::endl;
            bfsFile << "Number of vertices: " << tm->vertices.size() << std::endl;
            bfsFile << "Number of faces: " << tm->faceVector.size() << std::endl;
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

}