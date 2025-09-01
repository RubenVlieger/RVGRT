#ifndef CSDF_CUH
#define CSDF_CUH

#include "CArray.cuh"

// Define the dimensions of the coarse grid relative to the fine grid.
// A coarseness of 2 means each CSDF cell represents a 2x2x2 block of voxels.
#define COARSENESS 2
#define C_SIZEX (SIZEX / COARSENESS)
#define C_SIZEY (SIZEY / COARSENESS)
#define C_SIZEZ (SIZEZ / COARSENESS)
#define C_BYTESIZE (C_SIZEX * C_SIZEY * C_SIZEZ)

#define MAX_DIST 64

class CSDF {
public:
    CSDF();
    ~CSDF();

    // Allocates memory for the Coarse Signed Distance Field.
    void Allocate();

    // Generates the CSDF from a given fine-resolution bit array.
    // The `fineArray` is the input CArray containing your voxel data.
    void Generate(CArray& fineArray);

    // Provides access to the device pointer of the generated CSDF data.
    unsigned char* getPtr();

private:
    // CArray to hold the final CSDF data. Each element is one byte.
    CArray m_csdfArray;
};

#endif // CSDF_CUH
