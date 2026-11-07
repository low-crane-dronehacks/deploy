// Minimal reader for NumPy .npy / .npz files (np.save, np.savez, np.savez_compressed), no Python needed.
//
// An .npz is a zip archive of .npy members, stored or deflated (zip64 records handled). Supported .npy dtypes:
// little-endian / byte-order-free bool, int8..int64, uint8..uint64, float32, float64 and unicode strings (<U).
// Fortran-ordered arrays are rejected unless they are <= 1-D (where the order does not matter).
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace policynav {

struct NpyArray {
  char kind = 0;                 // numpy dtype kind: 'f', 'i', 'u', 'b' (bool), 'U' (unicode)
  std::size_t itemsize = 0;      // bytes per element (for 'U': 4 * number of characters)
  std::vector<std::size_t> shape;
  std::vector<std::uint8_t> data;  // raw little-endian element bytes, C order

  std::size_t size() const;      // number of elements (1 for a 0-d array)
  std::size_t ndim() const { return shape.size(); }

  // Element-wise conversions (throw on a non-numeric dtype). Exact for float32 -> float.
  std::vector<float> to_float() const;
  std::vector<double> to_double() const;
  std::vector<std::int64_t> to_int64() const;
  bool to_bool() const;          // scalar (size 1) bool / number -> bool
  std::string to_string() const;  // 0-d 'U' array -> UTF-8 (trailing NULs stripped)

  // Raw float32 view (throws unless the dtype is exactly float32).
  const float* f32() const;
};

// Parse one .npy blob (the bytes of an .npy file).
NpyArray parse_npy(const std::uint8_t* bytes, std::size_t n);
NpyArray load_npy(const std::string& path);

class NpzFile {
 public:
  NpzFile() = default;
  explicit NpzFile(const std::string& path);   // reads and decompresses every member eagerly

  bool has(const std::string& key) const { return arrays_.count(key) != 0; }
  const NpyArray& operator[](const std::string& key) const;   // key without the '.npy' suffix
  std::vector<std::string> keys() const;

 private:
  std::string path_;
  std::map<std::string, NpyArray> arrays_;
};

}  // namespace policynav
