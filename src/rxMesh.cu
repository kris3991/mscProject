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


//patches
void Patch::fillRibbons(preRxMeshDataStructure* rm)
{
	int t0{ 0 }, t1{ 0 }, t2{0};
	for (int i = 0; i < boundaryElements.size(); ++i)
	{
		t0 = rm->h_adjTriMap[boundaryElements[i]][0];
		t1 = rm->h_adjTriMap[boundaryElements[i]][1];
		t2 = rm->h_adjTriMap[boundaryElements[i]][2];
		ribbonElements.insert(t0);
		ribbonElements.insert(t1);
		ribbonElements.insert(t2);
	}
}

//RxMesh
RxMesh* RxMesh::rxMesh = nullptr;
RxMesh* RxMesh::GetInstance()
{
	if (rxMesh == nullptr) {
		rxMesh = new RxMesh();
	}
	return rxMesh;
}

RxMesh::RxMesh()
{

}


void RxMesh::calculateNormals(FileManager* fm, TriangleMesh* tm)
{
	//face data structure is 12 bytes of data.
	//8*3 + 12*3 =36 bytes of data.
	//i set shared memory size as 2 * 64 * 36 bytes.

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int threadCount = 2 << 6;
	int totalSize = faces.size();
	int blockCount = (totalSize + threadCount - 1) / threadCount;
	int normalCount = normals.size();
	cudaEventRecord(start);
	d_calculateNormals << <blockCount, threadCount >> > (d_faces, totalSize, d_normals, normalCount);
	cudaMemcpy(normals.data(), d_normals, sizeof(Vertex) * normals.size(), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float delta = 0;
	cudaEventElapsedTime(&delta, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	delta = delta * 1000;
	for (int i = 0; i < normals.size(); ++i)
	{
		tm->normals[i].x = normals[i].vx;
		tm->normals[i].y = normals[i].vy;
		tm->normals[i].z = normals[i].vz;
	}
	
	fm->writeNormals(tm, delta);


}


void RxMesh::clearCudaData()
{
	if(d_faces != nullptr)
		cudaFree(d_faces);
	if (d_normals != nullptr)
		cudaFree(d_normals);
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
	if (d_patchPositions != nullptr)
		cudaFree(d_patchPositions);
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
	//for global patchSize
	patchSize = (numFaces + patchCount - 1) / patchCount;
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
void d_populatePatchingArray(int* d_patchingArray, int size_N, int* d_adjascentTriangles)
{
	int tId = blockIdx.x * blockDim.x + threadIdx.x;
	if (tId < size_N)
	{
		//the idea is to check for the faces who have adjascent elements in a different patch.
		//store that in boundary.
		//populate based on adj triangles. so no invalid triangle pops up in the patch.
		//thread divergence better than o(n3)

		if (d_patchingArray[tId] != -1)
		{
			int patch = d_patchingArray[tId];
			int t0 = d_adjascentTriangles[tId * 3];
			int t1 = d_adjascentTriangles[tId * 3 + 1];
			int t2 = d_adjascentTriangles[tId * 3 + 2];

			if (t0 != -1)
			{
				atomicCAS(d_patchingArray + t0, -1, patch);
				//printf("t0 %d \t %d \n ", t0, d_patchingArray[t0]);
			}
			if (t1 != -1)
			{
				atomicCAS(d_patchingArray + t1, -1, patch);
				//printf("t1 %d \t %d \n ", t1, d_patchingArray[t1]);
			}
			if (t2 != -1)
			{
				atomicCAS(d_patchingArray + t2, -1, patch);
				//printf("t2 %d \t %d \n ", t2, d_patchingArray[t2]);
			}

		}
	}
}


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

	std::vector<int> test2(h_patchingArray.size(), -1);
	status = cudaMemcpy(test2.data(), d_patchingArray, sizeof(int) * h_patchingArray.size(), cudaMemcpyDeviceToHost); 

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
	if (cm->componentCount == 1)
	{
		h_initialiseSeedElements(tm, cm, pc);
	}
	else
	{
		h_initialiseSeedElementsMultiComp(tm, cm);
	}
	
	h_tempPatchArray.resize(h_patchingArray.size());
	std::fill(h_tempPatchArray.begin(), h_tempPatchArray.end(), -1);

	// i am putting 5 loops as convergence max.
	
	int loopCounter = 5;
	
	do{
		h_fillPatchingArrayWithSeedPoints();
		//clear the gpu values 

		int threadCount = patchSize;
		int blockCount = patchCount;
		int size_N = tm->faceVector.size() / 3;
		int sharedMemorySize = threadCount * sizeof(int);
		//set any random non zero value.
		int count = 0;
		int* d_count = 0;


		int* d_newPatchArray = 0;
		cudaMalloc(&d_newPatchArray, sizeof(int) * h_patchingArray.size());
		
		std::vector<int> indidualCounter(patchCount, 0);
		std::vector<int> prefixSum(patchCount, 0);
		std::vector<int> h_newPatchingArray(h_patchingArray.size(), 0);


		int* d_individualCounts = 0;
		cudaMalloc(&d_individualCounts, sizeof(int) * patchCount);
		cudaMemcpy(d_individualCounts, indidualCounter.data(), sizeof(int)* patchCount, cudaMemcpyHostToDevice);


		int* d_prefixSum = 0;
		cudaMalloc(&d_prefixSum, sizeof(int) * patchCount);

		cudaMalloc(&d_count, sizeof(int));
		cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
		//i was using blelloch earlier to get the sum of all face values.
		//but the reference in nvidea is for a single block.
		//my own code was not working.

		
		do
		{
			cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);
			d_populatePatchingArray << <blockCount, threadCount >> > (d_patchingArray, size_N, d_adjascentTriangles);
			cudaDeviceSynchronize();
			
			d_counter << <blockCount, threadCount >> > (d_patchingArray, size_N, d_count);
			cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
		} while (count != 0);

		cudaDeviceSynchronize();

	

		//grouping code
		//blelloch code given in nvidea works only for one block.
		
		d_computePatchCount << <blockCount, threadCount >> > (d_patchingArray, d_individualCounts, h_patchingArray.size());
		

		cudaMemcpy(indidualCounter.data(), d_individualCounts, patchCount * sizeof(int), cudaMemcpyDeviceToHost);

		prefixSum[0] = 0;
		for (int i = 1; i < patchCount; ++i) {
			prefixSum[i] = prefixSum[i - 1] + indidualCounter[i - 1];
		}

		cudaMemcpy(d_prefixSum, prefixSum.data(), sizeof(int) * patchCount, cudaMemcpyHostToDevice);
		cudaMemset(d_individualCounts, 0, patchCount * sizeof(int));

		d_arrangePatches << <blockCount, threadCount >> > (d_patchingArray, d_newPatchArray, d_individualCounts, d_prefixSum, h_patchingArray.size());

		cudaMemcpy(h_newPatchingArray.data(), d_newPatchArray, sizeof(int) * size_N, cudaMemcpyDeviceToHost);
		h_seedElements.clear();
		//select new seed points.
		int begin{ 0 };
		int end{ 0 };
		int temp{ 0 };

		//do this only for the last loop
		if(loopCounter == 1)
			cudaMemcpy(h_tempPatchArray.data(), d_patchingArray, sizeof(int) * size_N, cudaMemcpyDeviceToHost);

		for (int i = 0; i < h_newPatchingArray.size(); ++i)
		{
		//update seed step.
		// basically you update the seed points to make it centralised.	
			if (begin < patchSize)
			{
				begin = i * patchSize;
				end = i * patchSize + patchSize;

				if (cm->componentCount == 1)
				{
					int t0{ -1 }, t1{ -1 }, t2{ -1 };
					bool breakLoop = false;
					//safety.
					int count = 50;
					while (count && !breakLoop)
					{

						temp = *select_randomly(h_newPatchingArray.begin() + begin, h_newPatchingArray.begin() + end - 1);
						t0 = h_adjTriMap[temp][0];
						t1 = h_adjTriMap[temp][1];
						t2 = h_adjTriMap[temp][2];

						if (h_tempPatchArray[t0] == i && h_tempPatchArray[t1] == i && h_tempPatchArray[t2] == i)
						{
							breakLoop = true;
						}
						count--;
					}
					h_seedElements.push_back(temp);
				}
				else
				{
					h_initialiseSeedElementsMultiComp(tm, cm);
				}

			}
		}
		
		cudaMemset(d_patchingArray, -1, size_N * sizeof(int));
		cudaMemcpy(d_patchingArray, h_newPatchingArray.data(), sizeof(int) * h_newPatchingArray.size(), cudaMemcpyHostToDevice);
		std::fill(h_patchingArray.begin(), h_patchingArray.end(), -1);
	
	
		loopCounter--;
		cudaFree(d_prefixSum);
		cudaFree(d_newPatchArray);
		cudaFree(d_individualCounts);

	}while (loopCounter);

	cudaMemcpy(h_patchingArray.data(), d_patchingArray, sizeof(int)* h_patchingArray.size(), cudaMemcpyDeviceToHost);
	
}

void preRxMeshDataStructure::fillVertices(RxMesh* rMesh, TriangleMesh* tm)
{
	for (int i = 0; i < tm->vertices.size(); ++i)
	{
		Vertex v;
		v.vx = tm->vertices[i].x;
		v.vy = tm->vertices[i].y;
		v.vz = tm->vertices[i].z;
		rMesh->vertices.push_back(v);
	}
	//to fit in the shared memory without global fetch.
	for (int i = 0; i < tm->faceVector.size(); i += 3)
	{
		Faces f;
		f.v0 = tm->faceVector[i];
		f.v1 = tm->faceVector[i + 1];
		f.v2 = tm->faceVector[i + 2];

		f.v0PosX = rMesh->vertices[f.v0].vx;
		f.v0PosY = rMesh->vertices[f.v0].vy;
		f.v0PosZ = rMesh->vertices[f.v0].vz;

		f.v1PosX = rMesh->vertices[f.v1].vx;
		f.v1PosY = rMesh->vertices[f.v1].vy;
		f.v1PosZ = rMesh->vertices[f.v1].vz;

		f.v2PosX = rMesh->vertices[f.v2].vx;
		f.v2PosY = rMesh->vertices[f.v2].vy;
		f.v2PosZ = rMesh->vertices[f.v2].vz;
		rMesh->faces.push_back(f);
	}

	for (int i = 0; i < tm->normals.size(); ++i)
	{
		Vertex vn;
		vn.vx = tm->normals[i].x;
		vn.vy = tm->normals[i].y;
		vn.vz = tm->normals[i].z;
		rMesh->normals.push_back(vn);
	}

	int sizeofFace = sizeof(Faces);
	//faces.
	cudaMalloc(&rMesh->d_faces, sizeofFace * rMesh->faces.size());
	cudaMemcpy(rMesh->d_faces, rMesh->faces.data(), sizeofFace * rMesh->faces.size(), cudaMemcpyHostToDevice);
	//normals.
	cudaMalloc(&rMesh->d_normals, sizeof(Vertex) * rMesh->normals.size());
	cudaMemcpy(rMesh->d_normals, rMesh->normals.data(), sizeof(Vertex) * rMesh->normals.size(), cudaMemcpyHostToDevice);

}


void preRxMeshDataStructure::addRibbons(TriangleMesh* tm, RxMesh* rMesh)
{
	int size_N = tm->faceVector.size() / 3;
	
	std::vector<int> NegData(size_N, -1);

	cudaMemcpy(d_boundaryElements, NegData.data(), sizeof(int) * size_N, cudaMemcpyHostToDevice);

	int threadCount = patchSize;
	int blockCount = patchCount;

	cudaMalloc(&d_patchPositions, size_N * sizeof(int));
	cudaMemcpy(d_patchPositions, h_tempPatchArray.data(), size_N * sizeof(int), cudaMemcpyHostToDevice);

	d_findBoundaryPoints << <blockCount, threadCount >> > (d_patchingArray, size_N, d_boundaryElements, d_adjascentTriangles, d_patchPositions);
	cudaMemcpy(h_boundaryElements.data(), d_boundaryElements, sizeof(int) * size_N, cudaMemcpyDeviceToHost);

	rMesh->patchCount = patchCount;
	rMesh->patchSize = patchSize;
	rMesh->patches.resize(patchCount);

	int begin{ 0 }, end{ 0 }, patch{0};
	for (int i = 0; i < h_patchingArray.size(); ++i)
	{
		patch = (i / patchSize);
		begin = patch * patchSize;
		end = begin + patchSize;
		rMesh->patches[patch].faces.push_back(h_patchingArray[i]);
		if (h_boundaryElements[i] != -1)
		{
			rMesh->patches[h_boundaryElements[i]].boundaryElements.push_back(i);
		}
	}

	
	//for each patch add ribbon.
	for (int i = 0; i < rMesh->patches.size(); ++i)
	{
		rMesh->patches[i].fillRibbons(this);
	}

}


__global__ void d_computePatchCount(int* d_patchingArray, int* d_individualCounts, int size_N) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < size_N) {
		int value = d_patchingArray[tId];
		atomicAdd(&d_individualCounts[value], 1);
	}
}

__global__ void d_arrangePatches(int* d_patchingArray, int* d_newPatchArray, int* d_individualCounts, int* d_prefixSum, int size_N) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < size_N) {
		int value = d_patchingArray[tId];
		//printf("%d \n", value);
		int pos = atomicAdd(&d_individualCounts[value], 1);
		d_newPatchArray[d_prefixSum[value] + pos] = tId;
	}
}


__global__
void d_findBoundaryPoints(int* d_patchingArray, int size_N, int* d_boundaryElements, int* d_adjascentTriangles, int* d_patchPositions)
{
	int tId = blockDim.x * blockIdx.x + threadIdx.x;
	if (tId < size_N)
	{
		int current = d_patchingArray[tId];
		if (current != -1)
		{
			int begin = blockIdx.x * blockDim.x;
			int end = blockIdx.x * blockDim.x + blockDim.x;
			int t0{ -1 }, t1{ -1 }, t2{ -1 };
			t0 = d_adjascentTriangles[3 * current];
			t1 = d_adjascentTriangles[3 * current + 1];
			t2 = d_adjascentTriangles[3 * current + 2];
			if (t0 != -1 && d_patchPositions[t0] != blockIdx.x)
				d_boundaryElements[current] = blockIdx.x;
			if (t1 != -1 && d_patchPositions[t1] != blockIdx.x)
				d_boundaryElements[current] = blockIdx.x;
			if (t2 != -1 && d_patchPositions[t2] != blockIdx.x)
				d_boundaryElements[current] = blockIdx.x;
		}
		
	}

}

__global__
void d_calculateNormals(Faces* d_faces, int size_N, Vertex* d_normals, int normalCount)
{
	__shared__ Faces d_shared[128];
	int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < size_N)
	{
		d_shared[threadIdx.x] = d_faces[tId];

		float v0X = d_shared[threadIdx.x].v0PosX;
		float v0Y = d_shared[threadIdx.x].v0PosY;
		float v0Z = d_shared[threadIdx.x].v0PosZ;

		float v1X = d_shared[threadIdx.x].v1PosX;
		float v1Y = d_shared[threadIdx.x].v1PosY;
		float v1Z = d_shared[threadIdx.x].v1PosZ;

		float v2X = d_shared[threadIdx.x].v2PosX;
		float v2Y = d_shared[threadIdx.x].v2PosY;
		float v2Z = d_shared[threadIdx.x].v2PosZ;


		//vertex has same data as normal.
		Vertex v1_v0, v2_v0;
		v1_v0.vx = v1X - v0X;
		v1_v0.vy = v1Y - v0Y;
		v1_v0.vz = v1Z - v0Z;

		v2_v0.vx = v2X - v0X;
		v2_v0.vy = v2Y - v0Y;
		v2_v0.vz = v2Z - v0Z;

		//perform cross product
		Vertex vN;
		vN.vx = v1_v0.vy * v2_v0.vz - v1_v0.vz * v2_v0.vy;
		vN.vy = v1_v0.vz * v2_v0.vx - v1_v0.vx * v2_v0.vz;
		vN.vz = v1_v0.vx * v2_v0.vy - v1_v0.vy * v2_v0.vx;

		//we could add this at the beginning to skip a few iterations i guess.
		if (tId < normalCount)
		{
			atomicAdd(&d_normals[tId].vx, vN.vx);
			atomicAdd(&d_normals[tId].vy, vN.vy);
			atomicAdd(&d_normals[tId].vz, vN.vz);
		}


		
		//printf("vN.vx %f \t vN.vy %f \t vN.vz %f \n", vN.vx, vN.vy, vN.vz);
	}
}

