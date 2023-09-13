#include "../include/rxMesh.cuh"

//random uniform dist
//this code to select random point is referred from stack overflow.
//random selection for seed elements 
template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g) {
	std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
	std::advance(start, dis(g));
	return start;
}

template<typename Iter>
Iter select_randomly(Iter start, Iter end) {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	return select_randomly(start, end, gen);
}

preRxMeshDataStructure* preRxMeshDataStructure::rxMeshStruct = nullptr;


preRxMeshDataStructure* preRxMeshDataStructure::GetInstance()
{
	if (rxMeshStruct == nullptr) {
		rxMeshStruct = new preRxMeshDataStructure();
	}
	return rxMeshStruct;
}




preRxMeshDataStructure::preRxMeshDataStructure()
{
	d_faceVector = 0;
	d_adjascentTriangles = 0;
	h_adjascentTriangles.clear();
	sizeofFaceVector = 0;
	patchSize = 0;
	patchCount = 0;
	numFaces = 0;
}

void preRxMeshDataStructure::freeCudaData()
{
	if(d_adjascentTriangles != nullptr)
		cudaFree(d_adjascentTriangles);
	if(d_faceVector != nullptr)
		cudaFree(d_faceVector);
	if(d_adjascentTriangles != nullptr)
		cudaFree(d_adjascentTriangles);
	if (d_sizeN != nullptr)
		cudaFree(d_sizeN);
}

preRxMeshDataStructure::~preRxMeshDataStructure()
{

}

void preRxMeshDataStructure::initialise(TriangleMesh* tm)
{
	//in a manifold max number of adjascent triangles is 3. so adjascent 
	//tm->faceVector contains all the vertices in a face, dont confuse with h_faceIndexVector
	numFaces = tm->faceVector.size() / 3;
	int size_N = tm->faceVector.size();
	h_faceIndexVector.resize(numFaces);
	for (int i = 0; i < numFaces; ++i)
	{
		h_faceIndexVector[i] = i;
	}
	sizeofFaceVector = sizeof(int) * tm->faceVector.size();

	//initialise cuda data
	cudaError status;
	status = cudaMalloc(&d_adjascentTriangles, sizeofFaceVector);
	if (status != cudaSuccess)
	{
		std::cout << "error allocating d_adjascentTriangles" << std::endl;
	}
	status = cudaMalloc(&d_faceVector, sizeofFaceVector);
	if (status != cudaSuccess)
	{
		std::cout << "error allocating d_faceVector" << std::endl;
	}
	status = cudaMalloc(&d_adjascentTriangles, sizeofFaceVector);
	if (status != cudaSuccess)
	{
		std::cout << "error allocating d_adjascentTriangles" << std::endl;
	}
	//end

	//memcpy
	status = cudaMemcpy(d_faceVector, tm->faceVector.data(), sizeofFaceVector, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		std::cout << "memcpy failed for d_faceVector" << std::endl;
	}

	//this is for cudaAtomic operations.

	std::vector<int> temp(tm->faceVector.size(), -1);
	status = cudaMemcpy(d_adjascentTriangles, temp.data(), sizeofFaceVector, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		std::cout << "memcpy failed for d_adjascentTriangles" << std::endl;
	}
}

void preRxMeshDataStructure::h_initialiseSeedElements(TriangleMesh* tm, ComponentManager* cm, int pc)
{
	//if you create a patch bigger than the face count. the mesh will probably get messed up during patching.
	
	
	patchCount = pc;
	//each face has 3 elements]
	//last patch will have less elements depending on the patchCount;
	
	patchSize = (numFaces + patchCount - 1) / patchCount;
	int temp{ 0 };
	
	if (cm->componentCount == 1)
	{
		//o(n2) but n is small.
		
		int count{ 0 };
		int i = { 1 };
		int begin{ 0 }, end{ 0 };
		int temp{ 0 };
		while (count < patchCount)
		{
			end += patchSize;
			if (end > numFaces)
				end = numFaces;

			int next{ -1 }, previous{ -1 };
			while (next == -1 && previous == -1)
			{
				temp = *select_randomly(h_faceIndexVector.begin() + begin, h_faceIndexVector.begin() + end - 1);
				if (temp % 3 == 2)
				{
					next = h_adjascentTriangles[temp - 2];
					previous = h_adjascentTriangles[temp - 1];
				}
				else if (temp % 3 == 1)
				{
					next = h_adjascentTriangles[temp + 1];
					previous = h_adjascentTriangles[temp - 1];
				}
				else
				{
					next = h_adjascentTriangles[temp + 1];
					previous = h_adjascentTriangles[temp + 2];
				}
			}
			h_seedElements.push_back(temp);
			begin += patchSize;

			count++;
		}
			
	}

}

void preRxMeshDataStructure::h_initialiseSeedElementsMultiComp(TriangleMesh* tm, ComponentManager* cm)
{
	multiComponentPatchSize.clear();
	multiComponentPatchSize.resize(cm->componentCount);
	
	for (int i = 0; i < cm->componentCount; ++i)
	{
		int count{ 0 };
		int begin = cm->componentLocation[i], end = cm->componentLocation[i];
		int temp{ 0 };
		int start = cm->componentLocation[i];
		int stop = cm->componentLocation[i + 1];
		int faceCount = stop - start;
		multiComponentPatchSize[i] = (faceCount + multiComponentPatchCount[i] - 1) / multiComponentPatchCount[i];
		while (count < multiComponentPatchCount[i])
		{
			end += multiComponentPatchSize[i];
			if (end > numFaces)
				end = numFaces;

			int next{ -1 }, previous{ -1 };
			while (next == -1 && previous == -1)
			{
				temp = *select_randomly(h_faceIndexVector.begin() + begin, h_faceIndexVector.begin() + end - 1);
				if (temp % 3 == 2)
				{
					next = h_adjascentTriangles[temp - 2];
					previous = h_adjascentTriangles[temp - 1];
				}
				else if (temp % 3 == 1)
				{
					next = h_adjascentTriangles[temp + 1];
					previous = h_adjascentTriangles[temp - 1];
				}
				else
				{
					next = h_adjascentTriangles[temp + 1];
					previous = h_adjascentTriangles[temp + 2];
				}
			}
			h_seedElements.push_back(temp);
			begin += multiComponentPatchSize[i];

			count++;
		}
	}
}

void preRxMeshDataStructure::clear()
{
	h_seedElements.clear();
	h_adjascentTriangles.clear();
	multiComponentPatchCount.clear();
	multiComponentPatchSize.clear();
}

void preRxMeshDataStructure::clearSeedComponents()
{
	h_seedElements.clear();
	multiComponentPatchSize.clear();
}


void preRxMeshDataStructure::h_fillAdjascentTriangles(TriangleMesh* tm)
{
	clear();
	int size_N = tm->faceVector.size();
	int threadCount = 1 << 10;
	if (threadCount > size_N)
		threadCount = size_N;

	int gridSize = (size_N + threadCount - 1) / threadCount;
	h_adjascentTriangles.resize(size_N);
	int sharedMemorySize = 2 * threadCount;
	//in a manifold the max number of faces adjascent to one face is 3, but boundary vertices have -1 in this implementation.
	d_fillAdjascentTriangles << <gridSize, threadCount >> > (d_faceVector, d_adjascentTriangles, size_N);
	//copy the data for later operations.
	cudaMemcpy(h_adjascentTriangles.data(), d_adjascentTriangles, sizeofFaceVector, cudaMemcpyDeviceToHost);

}

__global__
void d_fillAdjascentTriangles(int* d_faceVector, int* d_adjascentTriangles, int size_N)
{
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	int lId = threadIdx.x;
	
	if (tId < size_N)
	{
		int v0 = d_faceVector[tId];
		int v1 = 0;
		if (tId % 3 == 2)
			v1 = d_faceVector[tId - 2];
		else
			v1 = d_faceVector[tId + 1];
		int v2, v3;
		for (int i = 0; i < size_N; ++i)
		{
			v2 = d_faceVector[i];
			if(i % 3 == 2)
				v3 = d_faceVector[i - 2];
			else
				v3 = d_faceVector[i + 1];
			if (v0 == v3 && v1 == v2)
			{
				d_adjascentTriangles[tId] = i/3;
			}
		}

	}
}