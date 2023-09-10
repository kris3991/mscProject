#include "../include/halfEdge.h"

//halfEdgeStruct::halfEdgeStruct(int v, int e, int n, int f, int otherEdge, bool b) :
//    vertex(v), edge(e), next(n), face(f), halfEdge(otherEdge), boundary(b) {};


//halfEdgeStruct::halfEdgeStruct(int v, Edge e, int n, int f, int otherEdge, bool b):
//    vertex(v), edge(e), next(n), face(f), halfEdge(otherEdge), boundary(b) {};

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
    halfEdges.clear();
   
}

void HalfEdge::initialise(TriangleMesh* tm)
{
	halfEdges.resize(tm->faceVector.size());
	std::fill(halfEdges.begin(), halfEdges.end(), -1);
	if (tm == nullptr)
		std::cout << "invalid mesh" << std::endl;
	else
	{
        //form the edgemap
        halfEdgeStruct he;
        he.boundary = false;
        he.edge = -1;
        he.face = -1;
        he.vertex = -1;
        he.halfEdge = -1;
        he.next = -1;

        for (int i = 0; i < tm->faceVector.size(); ++i)
        {
            int v0 = tm->faceVector[i];

            int v1;
            if (i % 3 == 2)
                v1 = tm->faceVector[i - 2];
            else
                v1 = tm->faceVector[i + 1];
            Edge e = Edge(v0, v1);
            edgeMap[i] = e;
            //initialise all halfedges as -1
            halfEdgeDs[i] = he;

        }

       

        //o(n2) for finding the half edge.
        int count0{ 0 };
        for (const auto& entry : edgeMap) {

            halfEdgeStruct he;
            he.boundary = false;
            he.edge = -1;
            he.face = -1;
            he.vertex = -1;
            he.halfEdge = -1;
            he.next = -1;


            Edge e = entry.second;
            int v0 = e.v0;
            int v1 = e.v1;

            int count1{ 0 };
            for (const auto& entry2 : edgeMap)
            {
                Edge e1 = entry2.second;
                int v2 = e1.v0;
                int v3 = e1.v1;

                
                if (v0 == v3 && v1 == v2 )
                {
                    if (halfEdgeDs[count0].halfEdge == -1)
                    {
                        /*halfEdgeDs[count0].vertex = v0;
                        halfEdgeDs[count0].boundary = false;
                        halfEdgeDs[count0].face = count0 / 3;
                        halfEdgeDs[count0].halfEdge = count1;
                        halfEdgeDs[count0].next = */
                    }
                }
                else
                {
                   
                    
                }
                count1++;
            }
            count0++;
        }
        //max query value is 
        query = std::to_string((tm->vertices.size()));

        //this is for queries.
		for (const auto& entry : halfEdgeDs) {
			int key = entry.first;                  
			const halfEdgeStruct& value = entry.second; 

			// Access the members of the halfEdgeStruct
			int edge = value.edge;
			int vertex = value.vertex;
			int halfEdge = value.halfEdge;
			int next = value.next;
			int face = value.face;
			//preprocessing for faster access later on.
            edgeHalfEdgeMap[edge] = value;
            faceHalfEdgeMap[face].push_back(value);
            vertexHalfEdgeMap[vertex].push_back(value);
		}
        int a = 30;
       
	}


}

void HalfEdge::adjascencyInformation(std::string fileName, TriangleMesh* tm)
{
    if (!fileName.size())
    {
        std::cout << "invalid file string name for adjascency information in half edge" << std::endl;
        return;
    }
    else
    {
       
        std::cout << "adjascency information:" << std::endl;
        for (int i = 0; i < halfEdges.size(); i += 3)
        {
            std::cout << "current face is " << i/3 << std::endl;
            std::cout << "vertices are" << std::endl;
            std::cout << tm->faceVector[i] << "\t" << tm->faceVector[i + 1] << "\t" << tm->faceVector[i + 2] << std::endl;
            int val;
            
            std::cout << "edges are: " << std::endl;
            for (int j = 0; j < 3; ++j)
            {
                val = 0;
                if ((i + j) % 3 == 2)
                    val = tm->faceVector[i + j - 2];
                else
                    val = tm->faceVector[i + j + 1];
                std::cout << i+j << "\t edge is:\t" << tm->faceVector[i + j] <<"\t"<< val << std::endl;
            }

            for (int k = 0; k < 3; ++k)
            {
                val = halfEdges[i + k];
                std::cout << "edge: \t" << i + k << std::endl;
                if (val == -1)
                {
                    std::cout << "border edge" << std::endl;
                }
                else
                {
                    std::cout << "half-edge: " << val << std::endl;
                    std::cout << "adjascent triangle is\t" << val / 3 << std::endl;
                }
            }
            int a = 30;
        }
    }
}



void HalfEdge::calculateVertexNormals(std::string fileName)
{
    if (!fileName.size())
    {
        std::cout << "invalid file string name for calculateVertexNormals in half edge" << std::endl;
        return;
    }
}


void HalfEdge::calculateOneRing(int vertex, TriangleMesh* tm)
{
    //basically the size of hashmap for each vertex
    std::cout << "vertex is " << vertex << std::endl;
    std::cout << "one ring count:" << edgeFaceMap[vertex].size() << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    //since one vertex is shared, loop through half edge data strucuture.
    //naive approach is to loop through the face array.
    std::set<int> oneRing;
    std::cout << "faces associated with this vertex: " << std::endl;
    
    int size = vertexHalfEdgeMap[vertex].size();
    if (size == 1)
        std::cout << "boundary vertex" << std::endl;
    else
    {
        std::cout << "number of connected faces:\t" << size << std::endl;
    }

    Edge currEdge, halfEdgePrev, nextEdge, halfEdgeNext, finalEdge, prevEdge;
    if (1)
    {
        for (int i = 0; i < size; ++i)
        {
            //boundary edges will be 0 or 1.
            
            {
                halfEdgeStruct hs = vertexHalfEdgeMap[vertex][i];
                //curr edge will always be added.
                if(hs.edge - 1 >= 0)
                    prevEdge = edgeMap[hs.edge - 1];
                currEdge = edgeMap[hs.edge];
                oneRing.insert(currEdge.v1);
                //check next edge
                nextEdge = edgeMap[hs.next];
                //check half edge.
                if(hs.halfEdge - 1 >= 0)
                    halfEdgePrev = edgeMap[hs.halfEdge - 1];
                //check next to half edge.
                //safety
                if(hs.halfEdge + 1 < edgeMap.size())
                    halfEdgeNext = edgeMap[hs.halfEdge + 1];
                //check final to curr edge.
                if(hs.next + 1 < edgeMap.size())
                    finalEdge = edgeMap[hs.next + 1];
                  int a = 30;
            }
            
            //halfEdgeStruct hs = vertexHalfEdgeMap[vertex][i];
            //std::cout << "face\t" << i << std::endl;
            //currEdge = edgeMap[hs.edge];
            //oneRing.insert(currEdge.v1);

            //int val = hs.edge % 3;
            ////curr map will always be added,
            //oneRing.insert(edgeMap[hs.edge].v1);
            //if (val == 0)
            //{
            //    nextEdge = edgeMap[hs.next];
            //    testEdge = edgeMap[hs.next + 1];
            //    int a = 30;
            //}

            //nextEdge = edgeMap[hs.next];
            //int v0 = nextEdge.v0;
            //int v1 = nextEdge.v1;
            //if (nextEdge.v0 == vertex)
            //    oneRing.insert(nextEdge.v1);
            //else if (nextEdge.v1 == vertex)
            //    oneRing.insert(nextEdge.v0);


            //testEdge = edgeMap[hs.edge - 1];
            ////both checks are not required, just in case
            //if (testEdge.v0 == vertex)
            //    oneRing.insert(testEdge.v1);
            //else if (testEdge.v1 == vertex)
            //    oneRing.insert(testEdge.v0);

        }
    }
    


}