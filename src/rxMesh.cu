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
	if (d_patchingArray != nullptr)
		cudaFree(d_patchingArray);
	if (d_boundaryElements != nullptr)
		cudaFree(d_boundaryElements);
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
	h_patchingArray.resize(numFaces);
	std::fill(h_patchingArray.begin(), h_patchingArray.end(), -1);



	h_boundaryElements.resize(numFaces);
	std::fill(h_boundaryElements.begin(), h_boundaryElements.end(), 0);

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

	//allocate patching array.
	status = cudaMalloc(&d_patchingArray, sizeof(int) * numFaces);
	if (status != cudaSuccess)
	{
		std::cout << "Patching allocation failed" << std::endl;
	}

	status = cudaMalloc(&d_boundaryElements, sizeof(int) * numFaces);
	if (status != cudaSuccess)
	{
		std::cout << "Patching allocation failed" << std::endl;
	}


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

			int t0{-1}, t1{ -1 }, t2{ -1 };
			while (t0 == -1 && t1 == -1 && t2 == -1)
			{
				temp = *select_randomly(h_faceIndexVector.begin() + begin, h_faceIndexVector.begin() + end - 1);
				t0 = h_adjTriMap[temp][0];
				t1 = h_adjTriMap[temp][2];
				t2 = h_adjTriMap[temp][2];
			}
			h_seedElements.push_back(temp);
			begin += patchSize;

			count++;
		}
			
	}
	//allocate seed array.
	cudaError status = cudaMalloc(&d_seedArray, sizeof(int) * h_seedElements.size());
	if (status != cudaSuccess)
	{
		std::cout << "allocation error for d_seedArray" << std::endl;
	}
	cudaMemcpy(d_seedArray, h_seedElements.data(), sizeof(int)* h_seedElements.size(), cudaMemcpyHostToDevice);
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

			int t0{ -1 }, t1{ -1 }, t2{-1};
			while (t0 == -1 && t1 == -1 && t2 == -1)
			{
				temp = *select_randomly(h_faceIndexVector.begin() + begin, h_faceIndexVector.begin() + end - 1);
				t0 = h_adjTriMap[temp][0];
				t1 = h_adjTriMap[temp][2];
				t2 = h_adjTriMap[temp][2];
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
	h_adjTriMap.clear();
}

void preRxMeshDataStructure::clearSeedComponents(TriangleMesh* tm)
{
	h_seedElements.clear();
	multiComponentPatchSize.clear();
	int size_N = tm->faceVector.size() / 3;
	h_patchingArray.resize(size_N);
	std::fill(h_patchingArray.begin(), h_patchingArray.end(), -1);
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

	//fill hash map for faster access.
	for (int i = 0; i < h_adjascentTriangles.size(); i = i + 3)
	{
		h_adjTriMap[i / 3] = { h_adjascentTriangles[i], h_adjascentTriangles[i + 1], h_adjascentTriangles[i + 2] };
	}
}

__global__
void d_fillAdjascentTriangles(int* d_faceVector, int* d_adjascentTriangles, int size_N)
{
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	int lId = threadIdx.x;
	//basic modulo operation for triangles.

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

__global__
void d_populatePatchingArray(int* d_patchingArray, int size_N, int* d_adjascentTriangles, int* d_boundaryElements)
{
	int tId = blockIdx.x * blockDim.x + threadIdx.x;
	if (tId < size_N)
	{
		//the idea is to check for the faces who have adjascent elements in a different patch.
		//store that in boundary.
		//populate based on adj triangles. so no invalid triangle pops up in the patch.
		//thread divergence better than o(n3)
		int begin = blockIdx.x * blockDim.x;
		int end = blockIdx.x* blockDim.x + blockDim.x;

		int t0, t1, t2;
		if (d_patchingArray[tId] != -1)
		{
			int patch = tId;
			t0 = d_adjascentTriangles[3 * tId];
			if (t0 > end || t0 < begin)
			{
				atomicCAS(d_boundaryElements + patch, 0, 1);
			}

			t1 = d_adjascentTriangles[3 * tId + 1];

			if (t1 > end || t1 < begin)
			{
				atomicCAS(d_boundaryElements + patch, 0, 1);
			}


			t2 = d_adjascentTriangles[3 * tId + 2];

			if (t2 > end || t2 < begin)
			{
				atomicCAS(d_boundaryElements + patch, 0, 1);
			}


			if (t0 != -1 && atomicCAS(d_patchingArray + t0, -1, blockIdx.x))
			{
				
			}
			if (t1 != -1 && atomicCAS(d_patchingArray + t1, -1, blockIdx.x))
			{
				
			}
			if (t2 != -1 && atomicCAS(d_patchingArray + t2, -1, blockIdx.x))
			{
				
			}
		}
	}
}

//__global__ void d_populatePatchingArray(int* d_patchingArray, int size_N, int* d_adjascentTriangles, int* d_count, bool* d_continue) {
//	int tId = blockIdx.x * blockDim.x + threadIdx.x;
//
//	while (*d_continue) {
//		if (tId < size_N) {
//			int t0, t1, t2;
//
//			if (d_patchingArray[tId] != -1) 
//			{
//				t0 = d_adjascentTriangles[3 * tId];
//				t1 = d_adjascentTriangles[3 * tId + 1];
//				t2 = d_adjascentTriangles[3 * tId + 2];
//
//				if (t0 != -1 && d_patchingArray[t0] == -1) {
//					d_patchingArray[t0] = t0;
//					*d_continue = true;
//				}
//				if (t1 != -1 && d_patchingArray[t1] == -1) {
//					d_patchingArray[t1] = t1;
//					*d_continue = true;
//				}
//				if (t2 != -1 && d_patchingArray[t2] == -1) {
//					d_patchingArray[t2] = t2;
//					*d_continue = true;
//				}
//			}
//		}
//		__syncthreads();
//
//		if (tId == 0) {
//			*d_continue = false;
//		}
//		__syncthreads();
//	}
//}



__global__
void d_counter(int* d_patchingArray, int size_N, int* d_count)
{
	int tId = blockIdx.x * blockDim.x + threadIdx.x;
	*d_count = 0;
	if (tId < size_N)
	{
		if (d_patchingArray[tId] == -1)
			atomicAdd(d_count, 1);
	}
}

void preRxMeshDataStructure::h_fillPatchingArrayWithSeedPoints()
{
	//there is no point in parallelising this block.
	//and will only be done once.
	for (int i = 0; i < h_seedElements.size(); ++i)
	{
		int currFace = h_seedElements[i];
		h_patchingArray[currFace] = i;
	}

	cudaError status = cudaMemcpy(d_patchingArray, h_patchingArray.data(), sizeof(int) * h_patchingArray.size(), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		std::cout << "memcpy failed for d_patchingArray" << std::endl;
	}

	//fill 0 for boundary just in case.
	status = cudaMemcpy(d_boundaryElements, h_boundaryElements.data(), sizeof(int) * h_patchingArray.size(), cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		std::cout << "memcpy failed for d_patchingArray" << std::endl;
	}

}

void preRxMeshDataStructure::h_populatePatches(TriangleMesh* tm, bool doIterations, ComponentManager* cm, int pc)
{

	//the algorithm involvest the following steps
	//initialise seed elements.
	//copy seed elements to patching array
	//for every non -1 element in patching array, add its neighbours.
	//keep counter to keep track of the patching process.
	//check if the faces are boundary. Select non boundary faces as seed for next itertaion.
	//repeat until 5th loop if iteration is enabled. 

	
	std::random_device rd;
	std::mt19937 gen(rd());


	clearSeedComponents(tm);
	h_initialiseSeedElements(tm, cm, pc);

	// i am putting 5 loops as convergence max.
	
	int loopCounter = 5;
	
	do{
		h_fillPatchingArrayWithSeedPoints();
		//clear the gpu values 
		cudaMemset(d_patchingArray, -1, tm->faceVector.size()/3);
		cudaMemset(d_boundaryElements, 0, tm->faceVector.size() / 3);

		int threadCount = patchSize;
		int blockCount = patchCount;
		int size_N = tm->faceVector.size() / 3;
		int sharedMemorySize = threadCount * sizeof(int);
		//set any random non zero value.
		int count = 0;
		int* d_count = 0;
		cudaMalloc(&d_count, sizeof(int));
		cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
		//i was using blelloch earlier to get the sum of all face values.
		//buts its easier to check for the number of -1s in the patching array
		/*for (int i = 0; i < 3; ++i)*/
		do
		{
			cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
			d_populatePatchingArray << <blockCount, threadCount >> > (d_patchingArray, size_N, d_adjascentTriangles, d_boundaryElements);
			cudaDeviceSynchronize();
			d_counter << <blockCount, threadCount >> > (d_patchingArray, size_N, d_count);
			cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
		} while (count != 0);

		
		cudaMemcpy(h_boundaryElements.data(), d_boundaryElements, sizeof(int) * size_N, cudaMemcpyDeviceToHost);

		//in the next iteration choose a seed point that is not a boundary element.
		//since i have chosen seed points which are not really boundary the following step could be possibly be skipped for more performance.
		//adding a condition to skip this.
		h_seedElements.clear();
		if (doIterations)
		{
			//nothing to parallelise here.
			for (int i = 0; i < patchCount; ++i)
			{
				int random_number;
				int begin = i * patchSize;
				int end = (i + 1) * patchSize;
				std::uniform_int_distribution<> distr(begin, end);
				int val = 1;
				//this loop will almost always be 1,for example in a patch(size 180) of a sphere obj of 960 faces, 10 are boundary around 0.02 percent in each patch.
				//while worst case is o(n3) here this will generally be o(1).
				while (val)
				{
					random_number = distr(gen);
					val = h_boundaryElements[random_number];
				}
				h_seedElements.push_back(random_number);
			}
			loopCounter--;
		}
		else
		{
			loopCounter = 0;
		}
		cudaFree(d_count);
		//clear all the cpu arrays.
		std::fill(h_patchingArray.begin(), h_patchingArray.end(), -1);
		std::fill(h_boundaryElements.begin(), h_boundaryElements.end(), 0);
	


	}while (loopCounter);

}



