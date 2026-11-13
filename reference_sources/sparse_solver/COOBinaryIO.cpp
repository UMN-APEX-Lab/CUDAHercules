#include "COOBinaryIO.h"
#include <zlib.h>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <fstream>

// Writes data and updates the CRC32
bool writeCRC(std::ofstream& file,
    const void* data,
    size_t bytes,
    uLong& crc)
{
    const char* p = static_cast<const char*>(data);

    while (bytes > 0)
    {
        uInt n = static_cast<uInt>(
            std::min<size_t>(bytes, 1024 * 1024));

        file.write(p, n);
        if (!file) return false;

        crc = crc32(crc,
            reinterpret_cast<const Bytef*>(p), n);

        p += n;
        bytes -= n;
    }

    return true;
}

// Reads data and updates the CRC32
bool readCRC(std::ifstream& file,
    void* data,
    size_t bytes,
    uLong& crc)
{
    char* p = static_cast<char*>(data);

    while (bytes > 0)
    {
        uInt n = static_cast<uInt>(
            std::min<size_t>(bytes, 1024 * 1024));

        file.read(p, n);
        if (!file) return false;

        crc = crc32(crc,
            reinterpret_cast<const Bytef*>(p), n);

        p += n;
        bytes -= n;
    }

    return true;
}

// ===================== A =====================
// Binary header: MatrixFileHeader {matrixInfo, rows, cols, nnz}
//   matrixInfo packs: sym, storeFormat=UPPER_TRIANGULAR, compressFormat=SPARSE_COO
//   rows == cols (square matrix)

bool readMatrixABinary(
    const std::string& infile,
    COOMatrix& mat,
    int& sym)
{
    std::ifstream ifile(
        infile.c_str(), std::ios::binary);

    if (!ifile.is_open())
        return false;

    // Initialize the CRC32
    uLong crc = crc32(0L, Z_NULL, 0);

    MatrixFileHeader header{};

    // Read the header and update the CRC
    if (!readCRC(
        ifile,
        &header,
        sizeof(MatrixFileHeader),
        crc))
        return false;

    int dataType, indexType, indexBase;
    int compressFormat, storeFormat, symType;

    parseMatrixInfo(
        header.matrixInfo,
        dataType,
        indexType,
        indexBase,
        compressFormat,
        storeFormat,
        symType);

    if (dataType != DT_COMPLEX128 ||
        indexType != IT_INT32 ||
        compressFormat != CF_SPARSE_COO)
    {
        return false;
    }

    // Recover the symmetry attribute
    if (symType == ST_SYM)
        sym = 2;
    else if (symType == ST_SYMPOS)
        sym = 1;
    else
        sym = 0;

    // Check the dimensions and nnz
    if (header.rows < 0 ||
        header.cols < 0 ||
        header.nnz < 0 ||
        header.rows > std::numeric_limits<int>::max() ||
        header.cols > std::numeric_limits<int>::max() ||
        header.nnz > std::numeric_limits<int>::max())
    {
        return false;
    }

    // Check the expected file size from the format
    // A corrupt header must not cause a huge allocation
    uint64_t nnz = static_cast<uint64_t>(header.nnz);

    constexpr uint64_t entryBytes =
        2 * sizeof(int32_t) + sizeof(Complex);

    const uint64_t maxSize =
        std::numeric_limits<uint64_t>::max();

    if (nnz > (maxSize - sizeof(MatrixFileHeader) - 4)
        / entryBytes)
        return false;

    uint64_t expectedSize =
        sizeof(MatrixFileHeader) +
        nnz * entryBytes +
        sizeof(uint32_t);

    ifile.seekg(0, std::ios::end);

    auto fileSize = ifile.tellg();

    if (fileSize < 0 ||
        static_cast<uint64_t>(fileSize) != expectedSize)
        return false;

    ifile.seekg(sizeof(MatrixFileHeader), std::ios::beg);

    // Read into a temporary so that a CRC failure leaves the caller's matrix untouched
    COOMatrix temp;

    temp.m_rows = static_cast<int>(header.rows);
    temp.m_cols = static_cast<int>(header.cols);
    temp.m_nnz = static_cast<int>(header.nnz);

    temp.row_indices.resize(temp.m_nnz);
    temp.col_indices.resize(temp.m_nnz);
    temp.data.resize(temp.m_nnz);

    // Read the row indices
    if (!readCRC(
        ifile,
        temp.row_indices.data(),
        nnz * sizeof(int32_t),
        crc))
        return false;

    // Read the column indices
    if (!readCRC(
        ifile,
        temp.col_indices.data(),
        nnz * sizeof(int32_t),
        crc))
        return false;

    // Read the complex values
    if (!readCRC(
        ifile,
        temp.data.data(),
        nnz * sizeof(Complex),
        crc))
        return false;

    // Read the CRC32 stored at the end of the file
    uint32_t storedCRC = 0;

    ifile.read(
        reinterpret_cast<char*>(&storedCRC),
        sizeof(storedCRC));

    if (!ifile)
        return false;

    // Verify the CRC32
    if (static_cast<uint32_t>(crc) != storedCRC)
    {
        // CRC mismatch
        return false;
    }

    // Update the matrix only after the check passed
    mat = std::move(temp);

    ifile.close();
    return true;
}

// ===================== B =====================
// Binary header: MatrixFileHeader {matrixInfo, rows, cols, nnz}
//   matrixInfo packs: nonsym, storeFormat=FULL, compressFormat=SPARSE_COO
//   cols == nports

bool readMatrixBBinary(const std::string& infile, COOMatrix& mat, int& nports)
{
    std::ifstream ifile(infile.c_str(), std::ios::binary);
    if (!ifile.is_open()) return false;

    MatrixFileHeader header;
    ifile.read(reinterpret_cast<char*>(&header), sizeof(MatrixFileHeader));

    int dataType, indexType, indexBase, compressFormat, storeFormat, symType;
    parseMatrixInfo(header.matrixInfo, dataType, indexType, indexBase,
                    compressFormat, storeFormat, symType);

    if (dataType != DT_COMPLEX128 || indexType != IT_INT32 ||
        compressFormat != CF_SPARSE_COO) {
        ifile.close();
        return false;
    }

    nports = static_cast<int>(header.cols);
    mat.m_rows = static_cast<int>(header.rows);
    mat.m_cols = nports;
    mat.m_nnz = static_cast<int>(header.nnz);

    mat.row_indices.resize(mat.m_nnz);
    mat.col_indices.resize(mat.m_nnz);
    mat.data.resize(mat.m_nnz);

    ifile.read(reinterpret_cast<char*>(mat.row_indices.data()), mat.m_nnz * sizeof(int32_t));
    ifile.read(reinterpret_cast<char*>(mat.col_indices.data()), mat.m_nnz * sizeof(int32_t));
    ifile.read(reinterpret_cast<char*>(mat.data.data()), mat.m_nnz * sizeof(Complex));

    ifile.close();
    return true;
}
