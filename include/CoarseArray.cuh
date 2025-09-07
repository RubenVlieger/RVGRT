#ifndef CSDF_CUH
#define CSDF_CUH

#include "CArray.cuh"

// Define the dimensions of the coarse grid relative to the fine grid.
// A coarseness of 2 means each CSDF cell represents a 2x2x2 block of voxels.
#define COARSENESSSDF 2
#define SDF_SIZEX (SIZEX / COARSENESSSDF)
#define SDF_SIZEY (SIZEY / COARSENESSSDF)
#define SDF_SIZEZ (SIZEZ / COARSENESSSDF)
#define SDF_BYTESIZE (SDF_SIZEX * SDF_SIZEY * SDF_SIZEZ)
#define SDF_MAX_DIST 64

#define COARSENESSGI 4
#define GI_SIZEX (SIZEX / COARSENESSGI)
#define GI_SIZEY (SIZEY / COARSENESSGI)
#define GI_SIZEZ (SIZEZ / COARSENESSGI)
#define GI_SIZE (GI_SIZEX * GI_SIZEY * GI_SIZEZ)
#define GI_BYTESIZE (GI_SIZEX * GI_SIZEY * GI_SIZEZ * sizeof(uchar4))


class CoarseArray {
public:
    CoarseArray();
    ~CoarseArray();

    // Allocates memory for the Coarse Signed Distance Field.
    void AllocateSDF();
    void AllocateGI();
    //void Allocate(const int byteSize);

    void GenerateSDF(CArray& fineArray);
    void GenerateGIdata(CArray& fineArray, CoarseArray csdf);
    void UpdateGI(CArray& fineArray, CoarseArray csdf);
    // Provides access to the device pointer of the generated data.
    unsigned char* getPtr();

private:
    // CArray to hold the data.
    CArray m_csdfArray;
};

#endif 