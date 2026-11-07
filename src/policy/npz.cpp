// .npy / .npz reader: zip central directory (+ zip64) -> stored / raw-deflate members (zlib) -> .npy header parse.
#include "policynav/npz.hpp"

#include <zlib.h>

#include <cctype>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

namespace policynav {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("npz: " + what); }

std::uint16_t rd16(const std::uint8_t* p) { return std::uint16_t(p[0] | (p[1] << 8)); }
std::uint32_t rd32(const std::uint8_t* p) {
  return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}
std::uint64_t rd64(const std::uint8_t* p) { return std::uint64_t(rd32(p)) | (std::uint64_t(rd32(p + 4)) << 32); }

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fail("cannot open " + path);
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// value of 'key' in the .npy header dict, e.g. "'<f4'", "False", "(5, 400, 640)"
std::string header_field(const std::string& h, const std::string& key) {
  auto k = h.find("'" + key + "'");
  if (k == std::string::npos) fail("npy header lacks '" + key + "': " + h);
  auto c = h.find(':', k);
  if (c == std::string::npos) fail("bad npy header: " + h);
  std::size_t i = c + 1;
  while (i < h.size() && h[i] == ' ') ++i;
  std::size_t j = i;
  if (h[i] == '(') {
    j = h.find(')', i);
    if (j == std::string::npos) fail("bad npy shape: " + h);
    ++j;
  } else if (h[i] == '\'' || h[i] == '"') {
    j = h.find(h[i], i + 1);
    if (j == std::string::npos) fail("bad npy header: " + h);
    ++j;
  } else {
    while (j < h.size() && h[j] != ',' && h[j] != '}') ++j;
  }
  return h.substr(i, j - i);
}

template <class T>
T load_le(const std::uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof(T));   // aarch64 / x86 are little-endian, as are the files we read
  return v;
}

template <class Out>
std::vector<Out> convert(const NpyArray& a) {
  const std::size_t n = a.size();
  std::vector<Out> out(n);
  const std::uint8_t* p = a.data.data();
  const std::size_t s = a.itemsize;
  for (std::size_t i = 0; i < n; ++i, p += s) {
    switch (a.kind) {
      case 'f':
        if (s == 4) out[i] = Out(load_le<float>(p));
        else if (s == 8) out[i] = Out(load_le<double>(p));
        else fail("unsupported float size");
        break;
      case 'i':
        if (s == 1) out[i] = Out(load_le<std::int8_t>(p));
        else if (s == 2) out[i] = Out(load_le<std::int16_t>(p));
        else if (s == 4) out[i] = Out(load_le<std::int32_t>(p));
        else out[i] = Out(load_le<std::int64_t>(p));
        break;
      case 'u':
        if (s == 1) out[i] = Out(load_le<std::uint8_t>(p));
        else if (s == 2) out[i] = Out(load_le<std::uint16_t>(p));
        else if (s == 4) out[i] = Out(load_le<std::uint32_t>(p));
        else out[i] = Out(load_le<std::uint64_t>(p));
        break;
      case 'b':
        out[i] = Out(p[0] != 0);
        break;
      default:
        fail(std::string("non-numeric dtype kind '") + a.kind + "'");
    }
  }
  return out;
}

std::vector<std::uint8_t> inflate_raw(const std::uint8_t* src, std::size_t n, std::size_t out_n) {
  std::vector<std::uint8_t> out(out_n);
  z_stream zs{};
  if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) fail("inflateInit2 failed");
  zs.next_in = const_cast<Bytef*>(src);
  zs.avail_in = static_cast<uInt>(n);
  zs.next_out = out.data();
  zs.avail_out = static_cast<uInt>(out_n);
  int rc = inflate(&zs, Z_FINISH);
  inflateEnd(&zs);
  if (rc != Z_STREAM_END || zs.total_out != out_n) fail("inflate failed (rc " + std::to_string(rc) + ")");
  return out;
}

}  // namespace

std::size_t NpyArray::size() const {
  std::size_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

std::vector<float> NpyArray::to_float() const { return convert<float>(*this); }
std::vector<double> NpyArray::to_double() const { return convert<double>(*this); }
std::vector<std::int64_t> NpyArray::to_int64() const { return convert<std::int64_t>(*this); }

bool NpyArray::to_bool() const {
  if (size() != 1) fail("to_bool on a non-scalar array");
  return convert<double>(*this)[0] != 0.0;
}

const float* NpyArray::f32() const {
  if (kind != 'f' || itemsize != 4) fail("array is not float32");
  return reinterpret_cast<const float*>(data.data());
}

std::string NpyArray::to_string() const {
  if (kind != 'U') fail("to_string on a non-unicode array");
  std::string s;
  for (std::size_t i = 0; i + 4 <= data.size(); i += 4) {
    std::uint32_t c = rd32(&data[i]);   // UTF-32LE -> UTF-8
    if (c == 0) break;
    if (c < 0x80) {
      s += char(c);
    } else if (c < 0x800) {
      s += char(0xC0 | (c >> 6));
      s += char(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
      s += char(0xE0 | (c >> 12));
      s += char(0x80 | ((c >> 6) & 0x3F));
      s += char(0x80 | (c & 0x3F));
    } else {
      s += char(0xF0 | (c >> 18));
      s += char(0x80 | ((c >> 12) & 0x3F));
      s += char(0x80 | ((c >> 6) & 0x3F));
      s += char(0x80 | (c & 0x3F));
    }
  }
  return s;
}

NpyArray parse_npy(const std::uint8_t* b, std::size_t n) {
  if (n < 10 || std::memcmp(b, "\x93NUMPY", 6) != 0) fail("not an .npy blob");
  const int major = b[6];
  std::size_t hlen, off;
  if (major == 1) {
    hlen = rd16(b + 8);
    off = 10;
  } else if (major == 2 || major == 3) {
    if (n < 12) fail("truncated .npy");
    hlen = rd32(b + 8);
    off = 12;
  } else {
    fail("unsupported .npy version " + std::to_string(major));
  }
  if (off + hlen > n) fail("truncated .npy header");
  const std::string h(reinterpret_cast<const char*>(b + off), hlen);

  NpyArray a;
  std::string descr = header_field(h, "descr");
  if (descr.size() < 4) fail("bad descr " + descr);
  descr = descr.substr(1, descr.size() - 2);           // strip quotes: '<f4'
  const char order = descr[0];
  if (order == '>') fail("big-endian arrays are not supported");
  a.kind = descr[1];
  const std::size_t count = std::stoul(descr.substr(2));
  a.itemsize = a.kind == 'U' ? 4 * count : count;
  if (a.kind != 'f' && a.kind != 'i' && a.kind != 'u' && a.kind != 'b' && a.kind != 'U') fail("unsupported dtype " + descr);

  std::string shp = header_field(h, "shape");          // "()", "(5,)", "(5, 400, 640)"
  for (std::size_t i = 1; i < shp.size();) {
    while (i < shp.size() && (shp[i] == ' ' || shp[i] == ',')) ++i;
    if (i >= shp.size() || shp[i] == ')') break;
    std::size_t j = i;
    while (j < shp.size() && std::isdigit(static_cast<unsigned char>(shp[j]))) ++j;
    if (j == i) fail("bad shape " + shp);
    a.shape.push_back(std::stoul(shp.substr(i, j - i)));
    i = j;
  }
  if (header_field(h, "fortran_order") == "True" && a.shape.size() > 1) fail("Fortran-ordered arrays are not supported");

  const std::size_t bytes = a.size() * a.itemsize;
  if (off + hlen + bytes > n) fail("truncated .npy data");
  a.data.assign(b + off + hlen, b + off + hlen + bytes);
  return a;
}

NpyArray load_npy(const std::string& path) {
  auto buf = read_file(path);
  return parse_npy(buf.data(), buf.size());
}

NpzFile::NpzFile(const std::string& path) : path_(path) {
  const auto buf = read_file(path);
  const std::size_t n = buf.size();
  const std::uint8_t* b = buf.data();
  if (n < 22) fail(path + ": too small for a zip archive");

  // End of central directory: scan back over a possible archive comment (<= 64 KiB).
  std::size_t eocd = std::string::npos;
  for (std::size_t i = n - 22 + 1; i-- > 0 && n - i <= 22 + 65535;) {
    if (rd32(b + i) == 0x06054b50) { eocd = i; break; }
  }
  if (eocd == std::string::npos) fail(path + ": no zip end-of-central-directory record");
  std::uint64_t entries = rd16(b + eocd + 10);
  std::uint64_t cd_off = rd32(b + eocd + 16);
  if ((entries == 0xFFFF || cd_off == 0xFFFFFFFFu) && eocd >= 20 && rd32(b + eocd - 20) == 0x07064b50) {
    const std::uint64_t z64 = rd64(b + eocd - 20 + 8);
    if (z64 + 56 > n || rd32(b + z64) != 0x06064b50) fail(path + ": bad zip64 end record");
    entries = rd64(b + z64 + 32);
    cd_off = rd64(b + z64 + 48);
  }

  std::size_t p = cd_off;
  for (std::uint64_t e = 0; e < entries; ++e) {
    if (p + 46 > n || rd32(b + p) != 0x02014b50) fail(path + ": bad central directory entry");
    const std::uint16_t method = rd16(b + p + 10);
    const std::uint32_t crc = rd32(b + p + 16);
    std::uint64_t csize = rd32(b + p + 20), usize = rd32(b + p + 24);
    const std::uint16_t name_len = rd16(b + p + 28), extra_len = rd16(b + p + 30), comment_len = rd16(b + p + 32);
    std::uint64_t local = rd32(b + p + 42);
    if (p + 46 + name_len + extra_len > n) fail(path + ": truncated central directory");
    std::string name(reinterpret_cast<const char*>(b + p + 46), name_len);
    // zip64 extended information: only the fields whose 32-bit value is saturated, in this order
    for (std::size_t x = p + 46 + name_len; x + 4 <= p + 46 + name_len + extra_len;) {
      const std::uint16_t id = rd16(b + x), sz = rd16(b + x + 2);
      if (id == 0x0001) {
        std::size_t f = x + 4;
        if (usize == 0xFFFFFFFFu) { usize = rd64(b + f); f += 8; }
        if (csize == 0xFFFFFFFFu) { csize = rd64(b + f); f += 8; }
        if (local == 0xFFFFFFFFu) { local = rd64(b + f); f += 8; }
      }
      x += 4 + sz;
    }
    p += 46 + name_len + extra_len + comment_len;

    if (local + 30 > n || rd32(b + local) != 0x04034b50) fail(path + ": bad local header for " + name);
    const std::size_t data_off = local + 30 + rd16(b + local + 26) + rd16(b + local + 28);
    if (data_off + csize > n) fail(path + ": truncated member " + name);
    std::vector<std::uint8_t> raw;
    if (method == 0) {
      raw.assign(b + data_off, b + data_off + csize);
    } else if (method == 8) {
      raw = inflate_raw(b + data_off, csize, usize);
    } else {
      fail(path + ": member " + name + " uses unsupported compression method " + std::to_string(method));
    }
    if (crc32(0L, raw.data(), static_cast<uInt>(raw.size())) != crc) fail(path + ": CRC mismatch in " + name);
    if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0) name.resize(name.size() - 4);
    arrays_.emplace(name, parse_npy(raw.data(), raw.size()));
  }
}

const NpyArray& NpzFile::operator[](const std::string& key) const {
  auto it = arrays_.find(key);
  if (it == arrays_.end()) fail(path_ + ": no array '" + key + "'");
  return it->second;
}

std::vector<std::string> NpzFile::keys() const {
  std::vector<std::string> k;
  for (const auto& kv : arrays_) k.push_back(kv.first);
  return k;
}

}  // namespace policynav
