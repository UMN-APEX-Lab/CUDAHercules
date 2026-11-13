#ifndef COO_BINARY_IO_H
#define COO_BINARY_IO_H

#include <complex>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>

typedef std::complex<double> Complex;

struct COOMatrix
{
    int m_rows = -1;
    int m_cols = -1;
    int m_nnz = -1;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<Complex> data;
};

// On-disk file header
struct MatrixFileHeader
{
    uint32_t matrixInfo;
    int64_t rows;
    int64_t cols;
    int64_t nnz;
};

// matrixInfo bit-field layout
static const int DATA_TYPE_MASK_POS        = 0;
static const int INDEX_TYPE_MASK_POS       = 4;
static const int INDEX_BASE_MASK_POS       = 6;
static const int COMPRESS_FORMAT_MASK_POS  = 8;
static const int STORE_FORMAT_MASK_POS     = 14;
static const int SYMMETRY_TYPE_MASK_POS    = 18;

static const uint32_t DATA_TYPE_MASK       = 0xFu  << DATA_TYPE_MASK_POS;
static const uint32_t INDEX_TYPE_MASK      = 0x3u  << INDEX_TYPE_MASK_POS;
static const uint32_t INDEX_BASE_MASK      = 0x3u  << INDEX_BASE_MASK_POS;
static const uint32_t COMPRESS_FORMAT_MASK = 0x1Fu << COMPRESS_FORMAT_MASK_POS;
static const uint32_t STORE_FORMAT_MASK    = 0x3u  << STORE_FORMAT_MASK_POS;
static const uint32_t SYMMETRY_TYPE_MASK   = 0xFu  << SYMMETRY_TYPE_MASK_POS;

inline uint32_t createMatrixInfo(int dataType, int indexType, int indexBase,
                                 int compressFormat, int storeFormat, int symType)
{
    return (static_cast<uint32_t>(dataType)       << DATA_TYPE_MASK_POS)  |
           (static_cast<uint32_t>(indexType)      << INDEX_TYPE_MASK_POS) |
           (static_cast<uint32_t>(indexBase)      << INDEX_BASE_MASK_POS) |
           (static_cast<uint32_t>(compressFormat) << COMPRESS_FORMAT_MASK_POS) |
           (static_cast<uint32_t>(storeFormat)    << STORE_FORMAT_MASK_POS) |
           (static_cast<uint32_t>(symType)        << SYMMETRY_TYPE_MASK_POS);
}

inline void parseMatrixInfo(uint32_t info, int& dataType, int& indexType, int& indexBase,
                            int& compressFormat, int& storeFormat, int& symType)
{
    dataType       = (info & DATA_TYPE_MASK)       >> DATA_TYPE_MASK_POS;
    indexType      = (info & INDEX_TYPE_MASK)      >> INDEX_TYPE_MASK_POS;
    indexBase      = (info & INDEX_BASE_MASK)      >> INDEX_BASE_MASK_POS;
    compressFormat = (info & COMPRESS_FORMAT_MASK) >> COMPRESS_FORMAT_MASK_POS;
    storeFormat    = (info & STORE_FORMAT_MASK)    >> STORE_FORMAT_MASK_POS;
    symType        = (info & SYMMETRY_TYPE_MASK)   >> SYMMETRY_TYPE_MASK_POS;
}

// Enum values of the matrixInfo fields
enum : int {
    DT_COMPLEX128 = 1 << 3,
    IT_INT32      = 1,
    IB_ZEROBASE   = 1,
    IB_ONEBASE    = 1 << 1,
    CF_SPARSE_COO = 1 << 4,
    SF_FULL       = 0,
    SF_UPPER      = 1 << 0,
    ST_NONSYM     = 0,
    ST_SYM        = 1 << 0,
    ST_SYMPOS     = 1 << 2,
};

// --- A: matrixInfo packs sym, storeFormat=UPPER, compressFormat=COO ---
bool readMatrixABinary(const std::string& infile, COOMatrix& mat, int& sym);

// --- B: matrixInfo packs nports in cols, storeFormat=FULL ---
bool readMatrixBBinary(const std::string& infile, COOMatrix& mat, int& nports);

#endif
